# =============================================================================
# 모듈 시작: eval_small_cnn.py — 작은 CNN(4층) 평가 전용 코드
# =============================================================================
# 목적: 학습은 하지 않고, 저장된 최고 모델(18 에폭)을 불러와 검증 데이터 전체를 예측한 뒤
#       팀원 step5_evaluate.py와 같은 형식으로 결과 파일 4개를 만든다.
#       (CNN_4layers.py를 import하면 학습이 다시 시작되므로 필요한 부분만 옮겨 옴)
# 실행: 프로젝트 폴더에서  python eval_small_cnn.py
#       (옵션 예: python eval_small_cnn.py --workers 4)
#       새 데이터(v2): python eval_small_cnn.py --csv-dir splits_csv_v2 --image-root division_224AFAD-X
#                      --weights results/small_cnn_v2_lr1e-3/small_cnn_best.pth --out-dir results/small_cnn_v2_lr1e-3/eval_val
#       (--csv-dir를 주면 그 폴더의 val.csv를, 안 주면 기존처럼 --data-dir 폴더(ImageFolder)를 평가)
# 출력: results/small_cnn_main/eval_val/
#       metrics.json, per_band.csv, confusion_matrix.png, predictions.csv
# =============================================================================

import argparse                                         # 명령줄 옵션(--weights 등)을 읽는 표준 라이브러리
import json                                             # metrics.json 저장용 (딕셔너리 → JSON 글자)
import os                                               # 폴더 만들기, 경로 다루기

import torch                                            # PyTorch 본체 (텐서, GPU 연산, 가중치 불러오기)
import torch.nn as nn                                   # nn: 신경망 층(Conv2d, BatchNorm2d, Linear 등)
from torch.utils.data import DataLoader, Dataset        # DataLoader: 배치로 묶기 / Dataset: CSV 모드용 사진 꺼내는 규칙
from PIL import Image                                   # CSV 모드에서 사진 파일 열기
from torchvision import datasets, transforms            # ImageFolder 데이터셋 / 이미지 전처리 도구

import pandas as pd                                     # 표(DataFrame) 만들기 → CSV 저장
import matplotlib                                       # 그래프 라이브러리 본체 (설정용)
matplotlib.use("Agg")                                   # (백엔드="Agg") 창을 띄우지 않고 파일로만 저장하는 모드
import matplotlib.pyplot as plt                         # 그림 그리기 도구
from sklearn.metrics import (                           # sklearn의 채점 함수들
    f1_score,                                           # macro-F1 계산
    precision_recall_fscore_support,                    # 나이대별 정밀도·재현율·F1·장수 계산
    confusion_matrix,                                   # 혼동 행렬 계산
)

# ---------- 설정값 (상수) ----------
# 아래 값들은 results/small_cnn_main/CNN_4layers_used.py(이 모델을 만든 코드)와 같아야 함
IMG_SIZE = 224                                          # 모델 입력 크기 (224×224). 출처: CNN_4layers_used.py 4-1
BATCH_SIZE = 64                                         # 한 번에 예측할 사진 수. 평가 결과에는 영향 없음 (eval 모드라 BN이 배치 통계를 안 씀)
NUM_CLASSES = 7                                         # 나이대 개수. 출처: CNN_4layers_used.py 4-2
BEST_EPOCH = 18                                         # small_cnn_best.pth가 저장된 에폭 (small_cnn_history.csv에서 val_macro_f1 최고)
MEAN = (0.485, 0.456, 0.406)                            # ImageNet 채널별 평균 (R, G, B). 출처: CNN_4layers_used.py 4-1
STD = (0.229, 0.224, 0.225)                             # ImageNet 채널별 표준편차 (R, G, B). 출처: CNN_4layers_used.py 4-1

# 팀원 common.py와 같은 나이대 이름 (번호 0~6 순서. ImageFolder 폴더 순서 '0 9' < '10 19' < ... < '60over'와 일치)
BAND_NAMES = ["0-9세", "10대", "20대", "30대", "40대", "50대", "60세 이상"]

# 기록값 (small_cnn_history.csv 18 에폭 행) — 재현 확인용
RECORDED_ACC = 0.5012428048142333                       # 18 에폭 val_acc
RECORDED_F1 = 0.4962069012723938                        # 18 에폭 val_macro_f1

# ---------- 검증용 transform ----------
# 출처: results/small_cnn_main/CNN_4layers_used.py 프로그램 4-1의 val_transform 그대로
# 학습 때 검증과 똑같은 조건이어야 기록값이 재현됨 (증강 없음)
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),            # (크기=(높이, 너비)) 비율 무시하고 224×224로 맞춤
    transforms.ToTensor(),                              # PIL 이미지(H, W, 3, 0~255) → 텐서(3, H, W, 0.0~1.0)
    transforms.Normalize(MEAN, STD),                    # (평균, 표준편차) 채널마다 (값 − 평균) / 표준편차
])


# ---------- CSV 데이터셋 (--csv-dir 모드) ----------
# CNN_4layers_unified.py의 FaceCSVDataset과 같은 방식 + samples 속성 (ImageFolder처럼 predictions.csv 경로용)
class CSVImageDataset(Dataset):
    def __init__(self, csv_path, image_root, transform):   # (csv_path=val.csv, image_root=path 기준 폴더, transform=전처리)
        df = pd.read_csv(csv_path)                      # CSV 읽기 (열: path, band, ...)
        self.samples = [(os.path.join(image_root, p), int(b)) for p, b in zip(df["path"], df["band"])]   # [(전체 경로, 나이대 번호), ...]
        self.transform = transform                      # 전처리 저장

    def __len__(self):                                  # 사진 수
        return len(self.samples)

    def __getitem__(self, i):                           # i번째 (이미지 텐서, 라벨)
        path, label = self.samples[i]                   # 경로, 정답
        return self.transform(Image.open(path).convert("RGB")), label   # RGB로 통일 후 전처리


# ---------- 모델 ----------
# 출처: results/small_cnn_main/CNN_4layers_used.py 프로그램 4-2의 SmallCNN 그대로
# (층 이름 conv1, bn1, ... fc 가 같아야 저장된 state_dict를 불러올 수 있음)
class SmallCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):        # (num_classes=출력 개수) 기본값 7
        super().__init__()                              # 부모 클래스(nn.Module) 초기화. 층 등록에 필요
        # 블록 1~4: Conv 3×3(크기 유지) → BN. 채널 3 → 32 → 64 → 128 → 256
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)     # (입력 3채널, 필터 32개, 3×3, 테두리 1칸, 편향 없음)
        self.bn1   = nn.BatchNorm2d(32)                                         # (채널 32) 정규화
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)    # (입력 32채널, 필터 64개)
        self.bn2   = nn.BatchNorm2d(64)                                         # (채널 64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)   # (입력 64채널, 필터 128개)
        self.bn3   = nn.BatchNorm2d(128)                                        # (채널 128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)  # (입력 128채널, 필터 256개)
        self.bn4   = nn.BatchNorm2d(256)                                        # (채널 256)
        self.pool    = nn.MaxPool2d(2, 2)               # (창문 2×2, 이동 2칸) 크기 절반
        self.gap     = nn.AdaptiveAvgPool2d((1, 1))     # (출력 크기 1×1) 채널마다 평균 1개
        self.dropout = nn.Dropout(p=0.3)                # (p=끌 확률) 학습 때만 작동, eval 모드에서는 꺼짐
        self.fc      = nn.Linear(256, num_classes)      # (입력 256개, 출력 7개) 나이대 점수

    def forward(self, x):                                     # x: (B, 3, 224, 224)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))   # (B, 32, 112, 112)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))   # (B, 64, 56, 56)
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))   # (B, 128, 28, 28)
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))   # (B, 256, 14, 14)
        x = self.gap(x)                                       # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)                             # (B, 256) 배치는 유지하고 한 줄로 펼침
        x = self.dropout(x)                                   # (B, 256) eval 모드라 그대로 통과
        x = self.fc(x)                                        # (B, 7) 나이대 7개 점수(logit)
        return x                                              # 7개 점수 반환


# ---------- 명령줄 옵션 ----------
def parse_args():                                        # 옵션을 읽어서 돌려주는 함수
    p = argparse.ArgumentParser(description="SmallCNN 검증 평가")   # (description=도움말 첫 줄) 옵션 해석기 생성
    p.add_argument("--weights", default="results/small_cnn_main/small_cnn_best.pth")  # (이름, default=안 주면 쓸 값) 가중치 파일
    p.add_argument("--data-dir", default="data/validation")                          # 평가할 이미지 폴더 (하위 폴더 = 나이대)
    p.add_argument("--out-dir", default="results/small_cnn_main/eval_val")           # 결과 저장 폴더
    p.add_argument("--csv-dir", default=None)            # 주면 이 폴더의 val.csv로 평가 (예: splits_csv_v2) / 안 주면 --data-dir 폴더 방식
    p.add_argument("--image-root", default=None)         # CSV path의 기준 폴더 (예: division_224AFAD-X). --csv-dir와 함께 사용
    p.add_argument("--split", default="val", choices=["val", "test"])   # CSV 모드에서 읽을 CSV (test는 최종 평가 때만 1회)
    p.add_argument("--workers", type=int, default=0)     # (type=정수로 변환) DataLoader 병렬 프로세스 수. 0 = 메인 프로세스만
    return p.parse_args()                                # 실제 명령줄을 읽어 args.weights, args.data_dir ... 로 돌려줌 (- 는 _ 로 바뀜)


# ---------- 전체 예측 수집 ----------
def predict_all(model, loader, device):                  # (model=평가할 모델, loader=검증 DataLoader, device=cpu/cuda)
    model.eval()                                         # 평가 모드: Dropout 끔, BN은 학습 때 저장된 평균·분산 사용
    all_preds, all_labels = [], []                       # 배치별 예측·정답을 모을 빈 리스트
    with torch.no_grad():                                # 기울기 계산 끔 → 메모리·시간 절약 (평가에는 역전파가 필요 없음)
        for images, labels in loader:                    # 배치 하나씩: images (B, 3, 224, 224), labels (B,)
            outputs = model(images.to(device))           # (이미지를 모델과 같은 장치로) → (B, 7) 점수
            all_preds.append(outputs.argmax(dim=1).cpu())  # (dim=1: 나이대 축) 가장 큰 점수의 번호 = 예측. sklearn용으로 CPU로
            all_labels.append(labels)                    # 정답은 원래 CPU에 있으므로 그대로
    return torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()   # 배치 리스트를 하나로 이어붙여 numpy 배열 (정답, 예측)으로


# ---------- 혼동 행렬 그림 ----------
def plot_confusion(cm, path):                            # (cm=7×7 개수 행렬, path=저장할 png 경로)
    plt.rcParams["font.family"] = "Malgun Gothic"        # 한글 나이대 이름이 깨지지 않도록 윈도우 기본 한글 폰트 지정
    plt.rcParams["axes.unicode_minus"] = False           # 한글 폰트에서 마이너스 기호가 깨지는 것 방지
    fig, ax = plt.subplots(figsize=(8, 7))               # (figsize=(가로, 세로) 인치) 그림 1개 생성
    im = ax.imshow(cm, cmap="Blues")                     # (행렬, cmap=색상표) 개수가 많을수록 진한 파랑
    fig.colorbar(im, ax=ax)                              # 오른쪽에 색 막대(개수 범례)
    ax.set_xticks(range(NUM_CLASSES))                    # x축 눈금 0~6
    ax.set_yticks(range(NUM_CLASSES))                    # y축 눈금 0~6
    ax.set_xticklabels(BAND_NAMES, rotation=45)          # (라벨, rotation=기울기) x축 = 예측 나이대
    ax.set_yticklabels(BAND_NAMES)                       # y축 = 정답 나이대
    ax.set_xlabel("예측")                                 # 열 = 모델이 고른 나이대
    ax.set_ylabel("정답")                                 # 행 = 실제 나이대
    ax.set_title(f"SmallCNN 혼동 행렬 (검증 {cm.sum():,}장)")   # 제목 (장수는 cm 합계 → 데이터에 따라 자동)
    thresh = cm.max() / 2                                # 칸 색이 진하면 흰 글씨, 연하면 검은 글씨로 쓰기 위한 기준
    for i in range(NUM_CLASSES):                         # i = 행 번호 (정답 나이대)
        for j in range(NUM_CLASSES):                     # j = 열 번호 (예측 나이대)
            ax.text(j, i, cm[i, j], ha="center", va="center",   # (x=열, y=행, 글자, 가운데 정렬) 칸 안에 개수 쓰기
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.tight_layout()                                   # 라벨이 잘리지 않게 여백 자동 조정
    fig.savefig(path, dpi=150)                           # (경로, dpi=해상도) png로 저장
    plt.close(fig)                                       # 메모리에서 그림 닫기


# =============================================================================
# 실행부 — 윈도우에서 num_workers > 0 이면 자식 프로세스가 이 파일을 다시 읽으므로
#          실행 코드는 전부 if __name__ == "__main__": 안에 둬야 무한 실행이 안 됨
# =============================================================================
if __name__ == "__main__":
    args = parse_args()                                  # 명령줄 옵션 읽기
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # GPU 있으면 cuda, 없으면 cpu
    print("using device:", device)                       # 확인용
    os.makedirs(args.out_dir, exist_ok=True)             # (폴더, exist_ok=이미 있어도 에러 안 냄) 결과 폴더 만들기

    # ---------- 데이터 ----------
    if args.csv_dir:                                     # CSV 모드 (v2): 라벨은 CSV의 band 열
        if not args.image_root:                          # path 기준 폴더가 없으면 사진을 못 찾음
            raise SystemExit("--csv-dir 를 쓰면 --image-root 도 주어야 합니다.")
        val_csv = os.path.join(args.csv_dir, f"{args.split}.csv")  # 평가 CSV 경로 (기본 val.csv)
        val_dataset = CSVImageDataset(val_csv, args.image_root, val_transform)   # (CSV, 기준 폴더, transform)
        print("CSV:", val_csv, "| image-root:", args.image_root)
    else:                                                # 기존 폴더 모드: 하위 폴더 이름 순서로 0~6 번호
        val_dataset = datasets.ImageFolder(args.data_dir, transform=val_transform)   # (폴더, transform)
        print("class_to_idx:", val_dataset.class_to_idx) # 라벨 순서 확인: {'0 9': 0, '10 19': 1, ..., '60over': 6} 이어야 함
    print("사진 수:", len(val_dataset))                    # 예전 15288 / v2 13790 이어야 함
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,   # shuffle=False: 순서 고정 → predictions.csv 행 = samples 순서
                            num_workers=args.workers)    # (num_workers=병렬 프로세스 수)

    # ---------- 모델 불러오기 ----------
    model = SmallCNN().to(device)                        # 빈 모델(무작위 가중치) 생성 후 장치로
    # torch.load(경로, map_location=장치): 저장된 가중치 딕셔너리를 읽어 지정 장치에 올림 (GPU 없는 PC에서도 열 수 있게)
    # model.load_state_dict(딕셔너리): 층 이름이 같은 자리에 가중치를 채워 넣음 (이름이 하나라도 다르면 에러)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    print("가중치:", args.weights)                          # 어떤 파일을 썼는지 기록

    # ---------- 예측 ----------
    y, p = predict_all(model, val_loader, device)        # y = 정답 번호 배열, p = 예측 번호 배열 (길이 15288)
    diff = p - y                                         # 예측 − 정답 (부호 있음). +면 나이를 많게, −면 적게 예측

    # ---------- 전체 지표 ----------
    acc = float((y == p).mean())                         # 정확도: (정답==예측)이 True인 비율
    # 한 칸 차이 정확도: 나이대 번호가 순서대로 붙어 있으므로(0-9세=0, 10대=1, ...)
    #   |예측 − 정답| ≤ 1 이면 "정답이거나 바로 옆 나이대" → 그 비율
    off1_acc = float((abs(diff) <= 1).mean())
    # macro-F1: 나이대마다 F1을 따로 구한 뒤 단순 평균 (사진 수가 적은 나이대도 똑같이 1/7 비중)
    #   labels=0~6: 혹시 한 번도 예측 안 된 나이대가 있어도 7개 모두 포함 / zero_division=0: 0으로 나누면 0점 처리
    macro_f1 = float(f1_score(y, p, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0))

    best_epoch = BEST_EPOCH                              # 기존 모델(18 에폭) 기본값
    cfg_path = os.path.join(os.path.dirname(args.weights), "config.json")   # 가중치 옆 config.json (unified 학습 코드가 저장)
    if args.csv_dir and os.path.exists(cfg_path):        # CSV 모드면 새 모델이므로 config.json의 best_epoch 사용
        with open(cfg_path, encoding="utf-8") as f:      # 읽기
            best_epoch = json.load(f).get("best_epoch")  # 최고 에폭 번호
    metrics = {"acc": acc, "off1_acc": off1_acc, "macro_f1": macro_f1,   # 팀원 metrics.json과 같은 키
               "n": int(len(y)), "best_epoch": best_epoch}
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:   # (경로, "w"=쓰기) 파일 열기
        json.dump(metrics, f, indent=2)                  # (딕셔너리, 파일, indent=들여쓰기 칸) 보기 좋게 저장

    # ---------- 나이대별 지표 (per_band.csv) ----------
    # precision_recall_fscore_support: 나이대마다
    #   정밀도 = 그 나이대라고 예측한 것 중 맞은 비율 (열 기준)
    #   재현율 = 실제 그 나이대인 사진 중 맞힌 비율 (행 기준) — 이것이 곧 "나이대별 정확도"
    #   F1    = 정밀도와 재현율의 조화평균
    #   장수  = 실제 그 나이대 사진 수 (support)
    prec, rec, f1, sup = precision_recall_fscore_support(y, p, labels=list(range(NUM_CLASSES)), zero_division=0)
    per_band = pd.DataFrame({"나이대": BAND_NAMES, "정밀도": prec, "재현율": rec, "F1": f1, "장수": sup}).round(3)  # 소수 셋째 자리 반올림
    per_band.to_csv(os.path.join(args.out_dir, "per_band.csv"), index=False, encoding="utf-8-sig")  # utf-8-sig: 엑셀에서 한글 안 깨짐

    # ---------- 혼동 행렬 (confusion_matrix.png) ----------
    # confusion_matrix(정답, 예측): 7×7 표. cm[i, j] = "정답이 i인데 j로 예측한 사진 수"
    #   대각선(i == j) = 맞힌 개수 / 대각선 바로 옆(|i − j| = 1) = 한 칸 빗나간 개수
    #   행 합계 = 그 나이대 실제 장수, 열 합계 = 그 나이대로 예측한 장수
    cm = confusion_matrix(y, p, labels=list(range(NUM_CLASSES)))
    plot_confusion(cm, os.path.join(args.out_dir, "confusion_matrix.png"))   # 그림 저장

    # ---------- 사진별 예측 (predictions.csv) ----------
    # val_dataset.samples: [(경로, 정답 번호), ...] — shuffle=False라 y, p와 같은 순서
    # os.path.relpath(경로): 실행 폴더 기준 상대 경로 / replace("\\", "/"): 윈도우 역슬래시를 슬래시로 통일
    paths = [os.path.relpath(s[0]).replace("\\", "/") for s in val_dataset.samples]
    pred_df = pd.DataFrame({
        "path": paths,                                   # 사진 경로 (예: data/validation/20 29/xxx.jpg)
        "band": y,                                       # 정답 나이대 번호
        "pred": p,                                       # 예측 나이대 번호
        "correct": (y == p).astype(int),                 # 맞으면 1, 틀리면 0
        "off1": (abs(diff) <= 1).astype(int),            # 한 칸 이내면 1
        "diff": diff,                                    # 예측 − 정답 (부호 있는 칸 수)
    })
    pred_df.to_csv(os.path.join(args.out_dir, "predictions.csv"), index=False, encoding="utf-8-sig")

    # ---------- 터미널 요약 ----------
    print(f"\n전체: acc {acc:.4f} | off1_acc {off1_acc:.4f} | macro_f1 {macro_f1:.4f} | n {len(y)}")
    print("\n나이대별 정확도(= 재현율):")
    for name, r, s in zip(BAND_NAMES, rec, sup):         # zip: 세 리스트를 같은 순서로 한 줄씩 묶어 꺼냄
        print(f"  {name:>7s}: {r:.3f}  ({s}장)")          # (:>7s = 7칸 오른쪽 정렬)
    wrong = diff != 0                                    # 틀린 사진 표시 (True/False 배열)
    near = (abs(diff) == 1) & wrong                      # 틀린 것 중 정확히 한 칸 옆 나이대
    print(f"\n오답 {wrong.sum()}장 중 옆 나이대(1칸) 오답: {near.sum()}장 ({near.sum() / wrong.sum():.1%})")

    # ---------- 기록값과 비교 (소수 셋째 자리) — 기존 모델(폴더 모드)일 때만 의미 있음 ----------
    if not args.csv_dir:                                 # CSV 모드(새 모델)는 비교할 기록값이 없어 생략
        ok_acc = round(acc, 3) == round(RECORDED_ACC, 3) # 반올림한 값이 같으면 True
        ok_f1 = round(macro_f1, 3) == round(RECORDED_F1, 3)
        print(f"\n기록값 비교: acc {acc:.4f} vs {RECORDED_ACC:.4f} → {'일치' if ok_acc else '불일치'} | "
              f"macro_f1 {macro_f1:.4f} vs {RECORDED_F1:.4f} → {'일치' if ok_f1 else '불일치'}")
    print("결과 저장:", args.out_dir)                      # 결과 폴더 위치 안내

# =============================================================================
# 모듈 끝: eval_small_cnn.py
# =============================================================================
