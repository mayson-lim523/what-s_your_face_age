# #############################################################################
#
#   작은 CNN(4층) — 팀 공통 학습 조건으로 다시 학습하기
#   인공지능 4조 · 작은 CNN 담당: LIM
#
#   [기존 CNN_4layers_latest.py와 다른 점] (팀원 ResNet-50·ConvNeXt와 같은 조건)
#     데이터     : ImageFolder → splits_csv/*.csv 를 읽는 Dataset
#     최적화     : Adam(lr 1e-3 고정) → AdamW(lr 1e-3 기본값, weight_decay 0.05) — 작은 CNN만 1e-3 (팀원 1e-4)
#     학습률     : 고정 → 첫 1 에폭 선형 워밍업 + 코사인 감소 (배치마다 갱신)
#     혼합 정밀도: 없음 → autocast + GradScaler
#     DataLoader : 학습용 drop_last=True, num_workers·pin_memory 사용
#     저장 위치  : 프로젝트 최상위 → --out-dir 폴더 (기본 results/small_cnn_unified)
#   그대로인 것: SmallCNN 구조, 증강·전처리, 나이대별 손실 가중치, 배치 64, 20 에폭, 시드 42,
#               최고 모델 기준(검증 macro-F1)
#
#   [모듈 구성]
#     U-1 설정·명령줄 옵션   U-2 데이터   U-3 모델   U-4 손실·최적화·스케줄
#     U-5 학습 루프          U-6 학습 곡선 그림
#     실행부는 맨 아래 if __name__ == "__main__": 안 (윈도우 num_workers > 0 대비)
#
#   [실행 방법] 프로젝트 폴더에서
#     본학습       : python models\CNN_4layers_unified.py
#     100장 점검   : python models\CNN_4layers_unified.py --subset 100 --epochs 100 --lr 1e-3 --out-dir results\check_subset100
#     2 에폭 점검  : python models\CNN_4layers_unified.py --stop-after 2 --out-dir results\check_full2
#     새 데이터(v2): python models\CNN_4layers_unified.py --csv-dir splits_csv_v2 --image-root division_224AFAD-X --out-dir results\small_cnn_v2_lr1e-3
#                    (v2 CSV의 path는 image-root 기준: training/0_9/xxx.jpg → train/val 모두 image-root를 사진 폴더로 사용)
#
# #############################################################################


# =============================================================================
#   ▶ [모듈 U-1 시작] 설정·명령줄 옵션
#   출력: 상수들, parse_args()
# =============================================================================

import argparse                                         # 명령줄 옵션(--epochs 등) 읽기
import json                                             # config.json 저장
import math                                             # 코사인(math.cos), 원주율(math.pi), NaN 검사(math.isnan)
import os                                               # 경로 합치기, 폴더 만들기
import time                                             # 에폭 시간 측정

import pandas as pd                                     # CSV 읽기·쓰기
import torch                                            # PyTorch 본체
import torch.nn as nn                                   # 신경망 층
from PIL import Image                                   # 사진 파일 열기 (Dataset에서 사용)
from torch.utils.data import Dataset, DataLoader        # Dataset: 사진 한 장 꺼내는 규칙 / DataLoader: 배치로 묶기
from torchvision import transforms                      # 전처리·증강
from sklearn.metrics import f1_score                    # macro-F1 (팀원과 같은 함수)
import matplotlib                                       # 그래프 설정용
matplotlib.use("Agg")                                   # 창 없이 파일로만 저장 (학습이 그림 창 때문에 멈추지 않게)
import matplotlib.pyplot as plt                         # 그래프 그리기

DATA_DIR = "data"                                       # 사진 폴더. CSV의 path는 이 폴더 기준 상대 경로 (예: data/training/20 29/xxx.jpg)
CSV_DIR = "splits_csv"                                  # train.csv / val.csv / test.csv 위치 (make_splits_csv.py로 생성)
IMG_SIZE = 224                                          # 입력 크기 224×224 (팀 조건 2)
BATCH_SIZE = 64                                         # 배치 크기 (팀 조건 6)
SEED = 42                                               # 난수 시드 (팀 조건 6)
NUM_CLASSES = 7                                         # 나이대 7개
WEIGHT_DECAY = 0.05                                     # AdamW 가중치 감쇠 (팀 조건 5)
TEAM_WEIGHTS = {                                        # 팀이 정한 나이대별 손실 가중치 — CSV 폴더 이름별 기준값 (계산값과 비교용, 팀 조건 4)
    "splits_csv":    [0.89, 0.85, 0.85, 0.85, 0.85, 1.51, 1.94],   # 예전 데이터 (학습 71,735장)
    "splits_csv_v2": [0.81, 1.11, 0.84, 0.84, 0.87, 1.36, 1.73],   # 새 데이터 v2 (학습 64,353장)
}
MEAN = (0.485, 0.456, 0.406)                            # ImageNet 채널별 평균 (R, G, B)
STD = (0.229, 0.224, 0.225)                             # ImageNet 채널별 표준편차 (R, G, B)


def parse_args():                                       # 명령줄 옵션을 읽어서 돌려줌
    p = argparse.ArgumentParser(description="SmallCNN 팀 공통 조건 학습")   # (description=도움말 첫 줄)
    p.add_argument("--epochs", type=int, default=20)    # (type=정수, default=20) 전체 에폭 수. 학습률 스케줄 길이도 이 값 기준
    p.add_argument("--lr", type=float, default=1e-3)    # 최대 학습률 (워밍업이 끝났을 때의 값). 작은 CNN만 1e-3 (팀원 모델은 1e-4)
    p.add_argument("--csv-dir", default=CSV_DIR)        # train.csv / val.csv 폴더 (기본 splits_csv, 새 데이터는 splits_csv_v2)
    p.add_argument("--image-root", default=None)        # 주면 train/val 모두 이 폴더 기준으로 path를 읽음 / 안 주면 DATA_DIR/training, DATA_DIR/validation
    p.add_argument("--no-augment", action="store_true") # (action=있으면 True) 학습 transform을 검증 transform과 같게 (증강 끔)
    p.add_argument("--workers", type=int, default=4)    # DataLoader 병렬 프로세스 수 (사진 읽기·증강을 GPU 계산과 동시에)
    p.add_argument("--out-dir", default="results/small_cnn_unified")   # 결과 저장 폴더
    p.add_argument("--subset", type=int, default=0)     # 0 = 전체 데이터 / N = 학습·검증 각 N장만, 증강 끔 (과적합 점검용)
    p.add_argument("--stop-after", type=int, default=0) # 0 = 끝까지 / K = K 에폭만 돌고 멈춤 (스케줄은 --epochs 기준 그대로)
    return p.parse_args()                               # args.epochs, args.out_dir ... 로 사용 (- 는 _ 로 바뀜)

# -----------------------------------------------------------------------------
# ◀ [모듈 U-1 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   ▶ [모듈 U-2 시작] 데이터
#   입력: splits_csv/train.csv, val.csv (열: path, band, ...)
#   출력: FaceCSVDataset, train_transform, val_transform
# =============================================================================

# 학습용 transform — CNN_4layers_latest.py 4-1과 같음 (팀 조건 3)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0), ratio=(0.95, 1.05)),  # (크기, scale=자를 면적 비율, ratio=가로세로 비율) 살짝 확대·이동
    transforms.RandomHorizontalFlip(p=0.5),             # (p=확률) 50% 좌우 반전
    transforms.RandomRotation(degrees=10),              # (degrees=최대 각도) −10° ~ +10° 회전
    transforms.ColorJitter(brightness=0.2, contrast=0.2),   # (밝기 폭, 대비 폭) 각각 0.8 ~ 1.2배
    transforms.ToTensor(),                              # PIL(H, W, 3, 0~255) → 텐서(3, H, W, 0~1)
    transforms.Normalize(MEAN, STD),                    # (평균, 표준편차) 채널마다 (값 − 평균) / 표준편차
])

# 검증용 transform — CNN_4layers_latest.py 4-1과 같음 (팀 조건 2, 증강 없음)
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),            # ((높이, 너비)) 224×224로
    transforms.ToTensor(),                              # 텐서로
    transforms.Normalize(MEAN, STD),                    # 정규화
])


# Dataset: "i번째 사진을 달라"고 하면 (이미지 텐서, 라벨)을 돌려주는 규칙
#   __len__    : 전체 사진 수 → DataLoader가 몇 배치인지 계산할 때 사용
#   __getitem__: i번째 사진을 그때그때 파일에서 읽어 transform 적용 (미리 다 읽지 않아 메모리 절약)
class FaceCSVDataset(Dataset):
    def __init__(self, csv_path, split_dir, transform):   # (csv_path=CSV 파일, split_dir=사진 폴더 예: data/training, transform=전처리)
        df = pd.read_csv(csv_path)                      # CSV 읽기 → 표
        self.paths = [os.path.join(split_dir, p) for p in df["path"]]   # 전체 경로 리스트 (data/training/20 29/xxx.jpg)
        self.labels = df["band"].astype(int).tolist()   # 나이대 번호 리스트 (0~6)
        self.transform = transform                      # 전처리 저장 (subset 모드에서 바꿀 수 있게 속성으로)

    def __len__(self):                                  # len(dataset) 이 호출하는 함수
        return len(self.paths)                          # 사진 수

    def __getitem__(self, i):                           # (i=사진 번호) dataset[i] 가 호출하는 함수
        img = Image.open(self.paths[i]).convert("RGB")  # 파일 열기 → RGB 3채널로 통일 (흑백·RGBA 사진 대비)
        return self.transform(img), self.labels[i]      # (전처리된 텐서, 라벨 번호)

# -----------------------------------------------------------------------------
# ◀ [모듈 U-2 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   ▶ [모듈 U-3 시작] 모델
#   출처: models/CNN_4layers_latest.py 프로그램 4-2의 SmallCNN 그대로 (구조·층 이름 동일)
#   구조: [Conv 3×3 → BN → ReLU → MaxPool] × 4 → GAP → Dropout 0.3 → Linear, 파라미터 390,695개
# =============================================================================

class SmallCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):        # (num_classes=출력 개수) 기본값 7
        super().__init__()                              # nn.Module 초기화 (층 등록에 필요)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)     # (입력 3채널, 필터 32개, 3×3, 테두리 1칸, 편향 없음) 블록 1
        self.bn1   = nn.BatchNorm2d(32)                                         # (채널 32) 정규화
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)    # (32 → 64) 블록 2
        self.bn2   = nn.BatchNorm2d(64)                                         # (채널 64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)   # (64 → 128) 블록 3
        self.bn3   = nn.BatchNorm2d(128)                                        # (채널 128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)  # (128 → 256) 블록 4
        self.bn4   = nn.BatchNorm2d(256)                                        # (채널 256)
        self.pool    = nn.MaxPool2d(2, 2)               # (창문 2×2, 이동 2칸) 크기 절반
        self.gap     = nn.AdaptiveAvgPool2d((1, 1))     # (출력 크기 1×1) 채널마다 평균 1개
        self.dropout = nn.Dropout(p=0.3)                # (p=끌 확률) 학습 때만 30% 끔
        self.fc      = nn.Linear(256, num_classes)      # (입력 256, 출력 7) 나이대 점수

    def forward(self, x):                                     # x: (B, 3, 224, 224)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))   # (B, 32, 112, 112)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))   # (B, 64, 56, 56)
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))   # (B, 128, 28, 28)
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))   # (B, 256, 14, 14)
        x = self.gap(x)                                       # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)                             # (B, 256) 펼치기
        x = self.dropout(x)                                   # (B, 256)
        x = self.fc(x)                                        # (B, 7) 점수(logit)
        return x                                              # 7개 점수 반환

# -----------------------------------------------------------------------------
# ◀ [모듈 U-3 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   ▶ [모듈 U-4 시작] 손실·최적화·학습률 스케줄
#   출력: class_weights(), make_lr_lambda()
# =============================================================================

def class_weights(train_csv, team):                    # (train_csv=학습 CSV 경로, team=팀 기준값 리스트 또는 None) → (가중치 텐서, 장수 텐서)
    labels = torch.tensor(pd.read_csv(train_csv)["band"].values)   # 학습 사진 전체의 라벨 (예전 71,735장 / v2 64,353장)
    counts = torch.bincount(labels, minlength=NUM_CLASSES).float()   # (라벨들, minlength=7) 나이대별 장수
    w = counts.sum() / (NUM_CLASSES * counts)           # 전체 / (7 × 나이대 장수) — 적은 나이대일수록 큰 가중치
    rounded = [round(v, 2) for v in w.tolist()]         # 소수 둘째 자리로 반올림
    print("나이대별 장수:", counts.int().tolist())
    print("계산한 가중치:", [round(v, 4) for v in w.tolist()])
    if team is None:                                    # 기준값이 없는 CSV 폴더 (TEAM_WEIGHTS에 없음)
        print("팀 가중치 비교: 기준값 없음 (계산값 그대로 사용)")
    elif rounded != team:                               # 기준값과 다르면 데이터가 바뀐 것 → 학습하지 않고 멈춤
        raise SystemExit(f"가중치가 팀 값과 다릅니다 {rounded} vs {team}. 멈춥니다.")
    else:                                               # 기준값과 같음
        print("팀 가중치 비교(둘째 자리): 일치")
    return w, counts                                    # 계산한 정확한 값 반환 (둘째 자리에서 자른 값이 아님) + 장수


def make_lr_lambda(total, warm):                        # (total=전체 배치 스텝 수, warm=워밍업 스텝 수) → 배율 함수
    # LambdaLR은 "실제 학습률 = 최대 학습률(--lr) × lr_lambda(step)" 으로 계산함. step은 0부터 1씩 증가 (배치마다)
    #
    # ① 워밍업 (step < warm, 첫 1 에폭):  배율 = (step + 1) / warm
    #    step 0 → 1/1005 ≈ 0.001 (거의 0), step 1004 → 1005/1005 = 1.0 (최대)  (v2 기준)
    #    → 처음부터 큰 학습률을 쓰면 무작위 가중치에서 크게 흔들리므로, 천천히 올려 안정적으로 출발
    # ② 코사인 감소 (step ≥ warm):  배율 = 0.5 × (1 + cos(π × 진행률)),  진행률 = (step − warm) / (total − warm)
    #    진행률 0 → cos 0 = 1 → 배율 1.0 / 진행률 0.5 → cos(π/2) = 0 → 0.5 / 진행률 1 → cos π = −1 → 0
    #    → 처음엔 천천히, 중간에 빠르게, 끝에서 다시 천천히 0으로 줄어듦 (끝부분에서 미세 조정)
    def lr_lambda(step):                                # (step=지금까지 scheduler.step()을 부른 횟수)
        if step < warm:                                 # 워밍업 구간
            return (step + 1) / warm                    # 선형 증가
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / (total - warm)))   # 코사인 감소
    return lr_lambda                                    # 함수 자체를 돌려줌 (LambdaLR에 넘김)

# -----------------------------------------------------------------------------
# ◀ [모듈 U-4 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   ▶ [모듈 U-5 시작] 학습 1 에폭 / 검증 1 에폭
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, use_amp):
    # (model=모델, loader=학습 로더, criterion=손실 함수, optimizer=AdamW, scheduler=LambdaLR,
    #  scaler=GradScaler, device=장치, use_amp=혼합 정밀도 사용 여부)
    model.train()                                       # 학습 모드: Dropout 켜짐, BN은 배치 통계 사용
    running_loss, correct, total = 0.0, 0, 0            # 손실 합계, 맞힌 수, 본 사진 수
    lr_start = optimizer.param_groups[0]["lr"]          # 이번 에폭 첫 배치에 쓸 학습률 (기록용)
    for images, labels in loader:                       # 배치 하나씩
        # non_blocking=True: pin_memory로 고정된 메모리에서 GPU로 복사할 때 기다리지 않고 다음 일을 진행 (속도 향상)
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        lr_used = optimizer.param_groups[0]["lr"]       # 이 배치에서 쓰는 학습률 (마지막 배치 값이 history의 lr)
        optimizer.zero_grad(set_to_none=True)           # 이전 기울기 지우기 (set_to_none=True: 0으로 채우는 대신 비워서 조금 빠름)

        # 혼합 정밀도(autocast): 이 블록 안의 계산 중 안전한 것(conv, linear)은 float16으로 해서
        #   GPU 계산이 빨라지고 메모리가 줄어듦. 정밀도가 필요한 계산(softmax, 손실 등)은 자동으로 float32 유지
        with torch.autocast(device_type=device.type, enabled=use_amp):   # (device_type=장치 종류, enabled=켤지 여부)
            outputs = model(images)                     # 순전파 (B, 7)
            loss = criterion(outputs, labels)           # 가중 교차 엔트로피 손실

        # GradScaler가 하는 일:
        #   float16은 표현할 수 있는 가장 작은 수가 커서, 아주 작은 기울기가 0으로 사라질 수 있음(underflow)
        #   ① scale(loss): 손실에 큰 수(예: 65536)를 곱한 뒤 backward → 기울기도 같이 커져서 0으로 안 사라짐
        #   ② step(optimizer): 가중치를 바꾸기 전에 기울기를 다시 그 수로 나눠 원래 크기로 되돌림
        #      이때 기울기에 inf/NaN이 있으면 이번 배치의 업데이트를 건너뜀 (가중치가 망가지는 것 방지)
        #   ③ update(): inf가 나왔으면 곱하는 수를 줄이고, 한동안 괜찮으면 다시 키움 (자동 조절)
        scaler.scale(loss).backward()                   # ① 키운 손실로 역전파
        scaler.step(optimizer)                          # ② 기울기 되돌리기 + 가중치 업데이트 (optimizer.step 대신)
        scaler.update()                                 # ③ 곱하는 수 조절
        scheduler.step()                                # 학습률 갱신 — 반드시 optimizer.step() "다음에" (배치마다, 팀 조건 5)

        running_loss += loss.item() * images.size(0)    # 배치 평균 × 장수 = 배치 합계
        correct += (outputs.argmax(dim=1) == labels).sum().item()   # 맞힌 수
        total += labels.size(0)                         # 본 사진 수
    return running_loss / total, correct / total, lr_start, lr_used   # (평균 손실, 정확도, 시작 lr, 마지막 lr)


def evaluate(model, loader, criterion, device, use_amp):    # (model, loader=검증 로더, criterion, device, use_amp)
    model.eval()                                        # 평가 모드: Dropout 끔, BN은 저장된 통계 사용
    running_loss, total = 0.0, 0                        # 손실 합계, 사진 수
    all_preds, all_labels = [], []                      # 전체 예측·정답 (F1은 전부 모아서 한 번에 계산해야 정확)
    with torch.no_grad():                               # 기울기 기록 끔
        for images, labels in loader:                   # 검증 배치 하나씩
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):   # 검증도 같은 혼합 정밀도로
                outputs = model(images)                 # 순전파
                loss = criterion(outputs, labels)       # 손실 (기록용)
            running_loss += loss.item() * images.size(0)    # 손실 합계
            total += labels.size(0)                     # 사진 수
            all_preds.append(outputs.argmax(dim=1).cpu())   # 예측 번호 (sklearn용으로 CPU)
            all_labels.append(labels.cpu())             # 정답
    y = torch.cat(all_labels)                           # (검증 장수,) 정답
    p = torch.cat(all_preds)                            # (검증 장수,) 예측
    acc = (p == y).float().mean().item()                # 정확도
    off1 = ((p - y).abs() <= 1).float().mean().item()   # 한 칸 차이 정확도: |예측 − 정답| ≤ 1 비율
    f1 = f1_score(y.numpy(), p.numpy(), average="macro",    # macro-F1: 나이대별 F1의 단순 평균
                  labels=list(range(NUM_CLASSES)), zero_division=0)   # 7개 모두 포함, 0 나눗셈은 0점
    return running_loss / total, acc, off1, f1          # (평균 손실, 정확도, off1, macro-F1)

# -----------------------------------------------------------------------------
# ◀ [모듈 U-5 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   ▶ [모듈 U-6 시작] 학습 곡선 그림
# =============================================================================

def plot_curves(df, path):                              # (df=history 표, path=저장 경로)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))     # (1행, 3열) 그래프 3칸
    axes[0].plot(df["epoch"], df["train_loss"], label="train")   # 학습 손실
    axes[0].plot(df["epoch"], df["val_loss"], label="val")       # 검증 손실
    axes[0].set_title("Loss")                           # 제목
    axes[1].plot(df["epoch"], df["train_acc"], label="train acc")    # 학습 정확도
    axes[1].plot(df["epoch"], df["val_acc"], label="val acc")        # 검증 정확도
    axes[1].plot(df["epoch"], df["val_macro_f1"], label="val macro-F1")   # 검증 macro-F1
    axes[1].set_title("Accuracy / F1")                  # 제목
    axes[2].plot(df["epoch"], df["lr"], marker="o")     # 에폭 끝 학습률
    axes[2].set_title("Learning rate (end of epoch)")   # 제목
    for ax in axes:                                     # 세 칸 공통 설정
        ax.set_xlabel("epoch")                          # x축 이름
        ax.legend() if ax is not axes[2] else None      # 범례 (lr 칸은 선이 하나라 생략)
    fig.tight_layout()                                  # 여백 자동 조정
    fig.savefig(path, dpi=150)                          # 저장
    plt.close(fig)                                      # 닫기

# -----------------------------------------------------------------------------
# ◀ [모듈 U-6 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   실행부 — 윈도우에서 num_workers > 0 이면 자식 프로세스가 이 파일을 다시 import 하므로
#            학습 코드는 전부 이 안에 있어야 무한 반복 실행이 안 됨
# =============================================================================
if __name__ == "__main__":
    args = parse_args()                                 # 명령줄 옵션
    torch.manual_seed(SEED)                             # 시드 고정 (모델 생성 "전에" → 초기 가중치·셔플·증강 고정)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 장치
    use_amp = device.type == "cuda"                     # 혼합 정밀도는 GPU에서만
    print("using device:", device, "| AMP:", use_amp)
    os.makedirs(args.out_dir, exist_ok=True)            # 결과 폴더 만들기

    # ---------- 데이터 ----------
    train_csv = os.path.join(args.csv_dir, "train.csv") # 학습 CSV 경로
    val_csv = os.path.join(args.csv_dir, "val.csv")     # 검증 CSV 경로
    if args.image_root:                                 # v2 방식: CSV path가 image-root 기준 (training/0_9/xxx.jpg)
        train_dir = val_dir = args.image_root           # train/val 모두 같은 최상위 폴더
    else:                                               # 예전 방식: CSV path가 분할 폴더 기준 (0 9/xxx.jpg)
        train_dir, val_dir = os.path.join(DATA_DIR, "training"), os.path.join(DATA_DIR, "validation")
    tr_tf = val_transform if args.no_augment else train_transform   # --no-augment면 학습도 증강 없이
    train_ds = FaceCSVDataset(train_csv, train_dir, tr_tf)          # 학습 (예전 71,735장 / v2 64,353장)
    val_ds = FaceCSVDataset(val_csv, val_dir, val_transform)        # 검증 (예전 15,288장 / v2 13,790장)
    print("CSV:", train_csv, "/", val_csv, "| image-root:", args.image_root or f"{DATA_DIR}/training, {DATA_DIR}/validation")
    print("증강:", "끔" if args.no_augment or args.subset else "켬")   # subset 모드는 아래에서 증강을 끔
    print("학습 transform:", tr_tf)                      # 실제 적용되는 학습 전처리 (--no-augment면 Resize·ToTensor·Normalize만)
    if args.subset:                                     # 과적합 점검 모드
        g = torch.Generator().manual_seed(SEED)         # 부분집합 뽑기 전용 난수 (매번 같은 N장)
        train_idx = torch.randperm(len(train_ds), generator=g)[:args.subset].tolist()   # 학습에서 무작위 N개 번호
        val_idx = torch.randperm(len(val_ds), generator=g)[:args.subset].tolist()       # 검증에서 무작위 N개 번호
        train_ds.transform = val_transform              # 증강 끄기 (N장을 "외울 수 있는지"만 보는 점검)
        train_ds = torch.utils.data.Subset(train_ds, train_idx)   # (원본, 번호 리스트) N장짜리 데이터셋
        val_ds = torch.utils.data.Subset(val_ds, val_idx)
    print("학습", len(train_ds), "장 / 검증", len(val_ds), "장")

    # drop_last=True: 마지막에 64장이 안 되는 자투리 배치를 버림
    #   v2: 64,353 = 64 × 1,005 + 33 → 매 에폭 33장을 버리고 1,005배치만 학습 (예전 71,735 = 64 × 1,120 + 55)
    #   (shuffle=True라 버려지는 33장은 에폭마다 다름 → 모든 사진이 골고루 쓰임)
    #   이유: 너무 작은 배치는 BatchNorm 통계가 불안정해짐 + 에폭당 스텝 수가 딱 맞아 스케줄 계산이 단순해짐 (팀원과 동일)
    #   subset 모드(100장)에서는 끔: 켜면 64장 한 배치만 학습하고 36장은 매번 버려짐
    loader_kw = dict(batch_size=BATCH_SIZE, num_workers=args.workers,   # 공통 옵션 (배치 크기, 병렬 프로세스 수)
                     pin_memory=True,                    # 고정(page-locked) 메모리에 배치를 올려 GPU 복사를 빠르게
                     persistent_workers=args.workers > 0)   # 에폭마다 worker를 새로 띄우지 않고 재사용 (윈도우에서 시작 시간 절약)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=not args.subset, **loader_kw)   # 학습: 섞기, 자투리 버림
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)    # 검증: 순서 고정, 전부 사용

    # ---------- 모델·손실·최적화·스케줄 ----------
    model = SmallCNN().to(device)                       # 모델 생성 → 장치로
    print("파라미터 수:", sum(p.numel() for p in model.parameters()))   # 390695 이어야 함
    team_w = TEAM_WEIGHTS.get(os.path.basename(os.path.normpath(args.csv_dir)))   # CSV 폴더 이름으로 팀 기준값 찾기 (없으면 None)
    weights, band_counts = class_weights(train_csv, team_w)   # 나이대별 가중치 (전체 학습 CSV 기준, 팀 기준값과 확인)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))   # (weight=나이대별 가중치) 가중 교차 엔트로피
    # AdamW: Adam과 같지만 weight_decay를 기울기와 섞지 않고 가중치에 직접 적용 (가중치를 매 스텝 조금씩 0쪽으로 당김 → 과적합 방지)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)   # (가중치들, lr=최대 학습률, weight_decay=감쇠)
    steps_per_epoch = len(train_loader)                 # 에폭당 배치 수 (v2 전체: 1,005 / 예전 1,120)
    total_steps = args.epochs * steps_per_epoch         # 전체 스텝 = 에폭 수 × 에폭당 배치 수 (--stop-after와 무관하게 --epochs 기준)
    warm_steps = steps_per_epoch                        # 워밍업 = 첫 1 에폭
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(total_steps, warm_steps))   # (optimizer, 배율 함수)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)   # ("cuda", enabled=GPU일 때만) 손실 크기 조절기
    print(f"에폭당 배치 {steps_per_epoch} | 전체 스텝 {total_steps} | 워밍업 {warm_steps}")

    # ---------- config.json (사용한 조건 전부) ----------
    config = {**vars(args),                             # vars(args): 명령줄 옵션 → 딕셔너리 (epochs, lr, workers, out_dir, subset, stop_after)
              "batch_size": BATCH_SIZE, "seed": SEED, "img_size": IMG_SIZE,
              "optimizer": "AdamW", "weight_decay": WEIGHT_DECAY,
              "schedule": "linear warmup 1 epoch + cosine, per batch",
              "amp": use_amp, "drop_last": not args.subset,
              "class_weights": [round(v, 4) for v in weights.tolist()],
              "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "warm_steps": warm_steps,
              "n_train": len(train_ds), "n_val": len(val_ds),
              "train_csv": train_csv, "val_csv": val_csv,  # 사용한 CSV 경로
              "train_dir": train_dir, "val_dir": val_dir,  # 실제 사진 폴더 (image-root 또는 DATA_DIR/...)
              "band_counts": band_counts.int().tolist(),  # 학습 CSV 나이대별 장수 (가중치 계산 근거)
              "augment": not (args.no_augment or args.subset),   # 학습 증강 사용 여부
              "best_epoch": None, "best_val_macro_f1": None}   # 학습 중 갱신
    config_path = os.path.join(args.out_dir, "config.json")    # 저장 경로

    def save_config():                                  # config를 파일에 쓰기 (최고 기록 갱신 때마다 다시 씀)
        with open(config_path, "w", encoding="utf-8") as f:    # 쓰기 모드로 열기
            json.dump(config, f, indent=2, ensure_ascii=False) # (딕셔너리, 파일, 들여쓰기, 한글 그대로)
    save_config()                                       # 학습 전 한 번 저장

    # ---------- 학습 루프 ----------
    history = []                                        # 에폭별 기록 (딕셔너리 리스트)
    best_f1 = -1.0                                      # 지금까지 최고 검증 macro-F1
    run_epochs = args.stop_after or args.epochs         # 실제로 돌릴 에폭 수 (stop-after가 0이면 전체)
    for epoch in range(1, run_epochs + 1):              # 1, 2, ..., run_epochs
        t0 = time.time()                                # 시작 시각
        tr_loss, tr_acc, lr_start, lr_end = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, use_amp)
        va_loss, va_acc, va_off1, va_f1 = evaluate(model, val_loader, criterion, device, use_amp)
        sec = time.time() - t0                          # 에폭 시간 (학습 + 검증)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,   # 팀원 history.csv 열 + lr, sec
                        "val_loss": va_loss, "val_acc": va_acc, "val_off1_acc": va_off1,
                        "val_macro_f1": va_f1, "lr": lr_end, "sec": sec})
        df = pd.DataFrame(history)                      # 표로
        df.to_csv(os.path.join(args.out_dir, "small_cnn_history.csv"), index=False)   # 매 에폭 저장 (중간에 멈춰도 기록 남음)

        mark = ""                                       # 최고 기록 표시
        if va_f1 > best_f1:                             # 검증 macro-F1 최고 갱신 (팀 기준)
            best_f1 = va_f1                             # 갱신
            torch.save(model.state_dict(), os.path.join(args.out_dir, "small_cnn_best.pth"))   # 가중치만 저장
            config["best_epoch"], config["best_val_macro_f1"] = epoch, va_f1   # 몇 에폭인지 기록
            save_config()                               # config.json 다시 쓰기
            mark = " ★ best"                            # 터미널 표시
        nan = " ⚠ NaN" if math.isnan(tr_loss) or math.isnan(va_loss) else ""   # 손실이 NaN이면 경고
        print(f"Epoch {epoch:3d}/{args.epochs} | lr {lr_start:.1e}→{lr_end:.1e} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} off1 {va_off1:.3f} F1 {va_f1:.3f} | {sec:.0f}초{mark}{nan}")

    plot_curves(pd.DataFrame(history), os.path.join(args.out_dir, "learning_curve.png"))   # 학습 곡선 저장
    print(f"best val macro-F1 {best_f1:.4f} (epoch {config['best_epoch']}) → {args.out_dir}")

# #############################################################################
# 모듈 끝: CNN_4layers_unified.py
# #############################################################################
