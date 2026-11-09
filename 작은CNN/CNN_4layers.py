#프로그램 4-1 얼굴 나이대 데이터 셋 구성하기
# =============================================================================
# 목적: data/ 폴더의 얼굴 이미지를 불러와서, 모델에 넣을 수 있는
#       (이미지 텐서, 나이대 번호) 묶음(배치)으로 만든다.
# =============================================================================

import torch                                            # PyTorch 본체 (텐서, GPU 연산, 시드 고정 등)
import os                                               # 운영체제 기능. 여기서는 경로를 합치는 os.path.join 사용
from torchvision import datasets, transforms            # datasets: ImageFolder 등 데이터셋 도구 / transforms: 이미지 전처리·증강 도구
from torch.utils.data import DataLoader, Subset         # DataLoader: 배치로 묶어 꺼내주는 도구 / Subset: 데이터셋 중 일부만 고르는 도구

# ---------- 설정값 (상수) ----------
DATA_DIR = "data"          # 데이터 폴더 이름. 프로젝트 폴더 기준 상대 경로 (프로젝트 폴더에서 실행해야 함)
IMG_SIZE = 224             # 모델에 넣을 이미지 한 변의 크기 (224×224 픽셀)
BATCH_SIZE = 64            # 한 번에 모델에 넣을 사진 수. 팀원 ResNet-50·ConvNeXt와 통일
SEED = 42                  # 난수 시드. 같은 시드면 매번 같은 무작위 결과가 나옴 (재현성)

torch.manual_seed(SEED)    # (시드값) PyTorch의 모든 무작위 동작(가중치 초기화, 셔플, 증강)을 고정.
                           # 모델을 만들기(4-2) "전에" 호출해야 초기 가중치까지 고정됨

# ImageNet 데이터셋의 채널별 평균·표준편차 (R, G, B 순서). 정규화와 4-5 시각화(되돌리기)에 사용
MEAN = (0.485,0.456,0.406 )    # (R 평균, G 평균, B 평균)
STD = ( 0.229, 0.224, 0.225)   # (R 표준편차, G 표준편차, B 표준편차)

# (조건이 참일 때 값 if 조건 else 거짓일 때 값)
# torch.cuda.is_available(): GPU를 쓸 수 있으면 True → "cuda", 아니면 "cpu"
# torch.device(...)는 "어떤 장치를 쓸지" 이름표만 만든다. 실제 이동은 .to(device)로 따로 해야 함
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)                          # 확인용: using device: cuda 가 나와야 GPU 사용

# ---------- 학습용 transform: 크기 맞추기 → 증강 → 텐서 → 정규화 ----------
# transforms.Compose([...]): 리스트 안의 변환을 위에서부터 순서대로 적용
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),                        # (크기) 숫자 하나 → 짧은 변을 224로 맞추고 비율은 유지. 튜플 (224, 224)를 주면 강제로 늘려서 얼굴이 찌그러질 수 있음
    transforms.CenterCrop(IMG_SIZE),                    # (크기) 가운데를 224×224로 자름. 지금 데이터는 이미 224라 안전장치 역할
    transforms.RandomHorizontalFlip(p=0.5),             # (p=확률) 50% 확률로 좌우 반전
    transforms.RandomRotation(degrees=10),              # (degrees=최대 각도) -10° ~ +10° 사이에서 무작위 회전. 빈 모서리는 검은색으로 채워짐
    transforms.ColorJitter(brightness=0.2),             # (brightness=변화 폭) 밝기를 0.8배 ~ 1.2배 사이에서 무작위로 바꿈
    transforms.ToTensor(),                              # PIL 이미지(H, W, 3, 0~255) → 텐서(3, H, W, 0.0~1.0)로 변환. 채널이 맨 앞으로 옴
    transforms.Normalize(MEAN, STD),                    # (평균, 표준편차) 채널마다 (값 − 평균) / 표준편차 → 대략 평균 0, 표준편차 1로 맞춤
    ])

# 검증용: 증강 없이 크기 맞추기 + 텐서 + 정규화만
# 검증은 매번 같은 조건으로 채점해야 하므로 무작위 증강을 넣지 않는다
val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),                        # (크기) 짧은 변을 224로 (비율 유지)
    transforms.CenterCrop(IMG_SIZE),                    # (크기) 가운데 224×224
    transforms.ToTensor(),                              # PIL 이미지 → 텐서 (0~1)
    transforms.Normalize(MEAN, STD),                    # (평균, 표준편차) 학습용과 같은 값으로 정규화
])

# ---------- dataset ----------
# datasets.ImageFolder(root=폴더, transform=변환)
#   root 안의 하위 폴더 이름이 곧 라벨이 됨 (폴더 이름을 문자열 순서로 정렬해서 0, 1, 2 ... 번호를 붙임)
#   transform: 사진을 꺼낼 때마다 적용할 전처리
train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "training"),            # os.path.join(기준 폴더, 하위 폴더) → "data\training". 학습 데이터 71,735장
    transform=train_transform,                          # 학습용 transform (증강 포함)
)
val_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "validation"),          # "data\validation". 검증 데이터 15,288장 (testing 폴더는 최종 평가 때 한 번만 사용)
    transform=val_transform,                            # 검증용 transform (증강 없음)
)

print(train_dataset.class_to_idx)          # 라벨 순서 확인: {'0 9': 0, '10 19': 1, ..., '60over': 6} (폴더 이름 → 번호)
print(len(train_dataset), len(val_dataset))  # len(데이터셋) = 사진 수. 71735 15288 이어야 정상

# ---------- 소규모 부분집합 ----------
N_TRAIN = 100      # 1단계: 100장 과적합 테스트 → 나중에 늘리기
N_VAL   = 100      # 검증에 쓸 사진 수 (본 학습 때는 부분집합 없이 validation 전체를 써야 함)

# torch.Generator().manual_seed(시드): 이 부분집합 뽑기 전용 난수 생성기를 만들고 시드 고정
g = torch.Generator().manual_seed(SEED)    # 시드 고정 → 매번 같은 부분집합
# torch.randperm(n, generator=g): 0 ~ n-1 을 무작위로 섞은 순서 (예: [5021, 13, 70002, ...])
#   [:N_TRAIN]: 그중 앞에서 N_TRAIN개만 / .tolist(): 텐서 → 파이썬 리스트
#   ImageFolder는 클래스 순서로 정렬돼 있어서, range(100)처럼 앞에서 자르면 0~9세만 뽑힘 → 섞어서 뽑아야 함
train_idx = torch.randperm(len(train_dataset), generator=g)[:N_TRAIN].tolist()
val_idx   = torch.randperm(len(val_dataset), generator=g)[:N_VAL].tolist()

# Subset(원본 데이터셋, 인덱스 리스트): 원본 중 해당 번호의 사진만 쓰는 데이터셋
train_subset = Subset(train_dataset,train_idx)
val_subset   = Subset(val_dataset,val_idx)

# ---------- dataloader ----------
# DataLoader(데이터셋, batch_size=배치 크기, shuffle=섞을지 여부)
#   for문으로 꺼낼 때마다 (이미지 배치, 라벨 배치)를 돌려줌
#   shuffle=True: 에폭마다 순서를 섞음 (학습용) / False: 항상 같은 순서 (검증용)
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False)

# 확인용: 첫 배치 하나만 꺼내 보기
# iter(로더): 꺼낼 수 있는 상태로 만들기 / next(...): 하나 꺼내기
images, labels = next(iter(train_loader))
print(images.shape)   # torch.Size([64, 3, 224, 224]) 이어야 정상 → (배치, 채널, 높이, 너비)
print(labels[:10])    # 앞에서 10개 라벨. 0~6 사이 숫자들이 섞여 나와야 정상
print(images.mean().item(), images.std().item())  # 전체 평균·표준편차. 정규화됐으면 대략 0 근처, 1 근처 (.item(): 텐서 → 파이썬 숫자)

# 프로그램 4-2 4층 CNN 모델 구성하기
# =============================================================================
# 구조: [Conv 3×3 → BN → ReLU → MaxPool] 블록 4개 → GAP → Dropout → Linear
#       블록마다 크기는 절반(224→112→56→28→14), 채널은 2배(32→64→128→256)
#       파라미터 수: 390,695개
# =============================================================================

import torch.nn as nn       # nn: 신경망 층(Conv2d, BatchNorm2d, Linear 등)이 들어 있는 모듈

NUM_CLASSES = 7             # 분류할 나이대 개수 (0~9, 10대, 20대, 30대, 40대, 50대, 60세 이상)

# nn.Module을 상속해서 모델을 정의. __init__에서 층을 만들고, forward에서 데이터가 지나가는 순서를 정함
class SmallCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):        # (num_classes=출력 개수) 기본값 7
        super().__init__()                              # 부모 클래스(nn.Module)의 초기화. 층을 등록하려면 반드시 필요

        # nn.Conv2d(입력 채널, 출력 채널, kernel_size=필터 크기, padding=테두리 칸 수, bias=편향 사용 여부)
        #   출력 채널 = 필터 개수 = 찾을 패턴의 종류 수
        #   출력 크기 = (H + 2×padding − kernel_size) / stride + 1 → (224 + 2 − 3) / 1 + 1 = 224 (크기 유지)
        #   bias=False: 바로 뒤 BatchNorm이 평균을 빼므로 bias가 의미 없음
        # nn.BatchNorm2d(채널 수): 채널마다 평균 0, 분산 1로 정규화. 채널마다 γ, β 2개를 학습 → 파라미터 2 × 채널
        #   바로 앞 conv의 출력 채널 수와 같아야 함

        # 블록 1: 3 → 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)     # (RGB 3채널 입력, 필터 32개, 3×3, 테두리 1칸) 파라미터 3×32×3×3 = 864
        self.bn1   = nn.BatchNorm2d(32)                                         # (채널 32) 파라미터 64
        # 블록 2: 32 → 64
        self.conv2 = nn.Conv2d(32, 64,kernel_size=3, padding=1, bias=False)     # (32채널 입력, 필터 64개) 파라미터 32×64×9 = 18,432
        self.bn2   = nn.BatchNorm2d(64)                                         # (채널 64) 파라미터 128
        # 블록 3: 64 → 128
        self.conv3 = nn.Conv2d(64, 128,kernel_size=3, padding=1, bias=False)    # (64채널 입력, 필터 128개) 파라미터 64×128×9 = 73,728
        self.bn3   = nn.BatchNorm2d(128)                                        # (채널 128) 파라미터 256
        # 블록 4: 128 → 256
        self.conv4 = nn.Conv2d(128, 256,kernel_size=3, padding=1, bias=False)   # (128채널 입력, 필터 256개) 파라미터 128×256×9 = 294,912 (전체의 약 75%)
        self.bn4   = nn.BatchNorm2d(256)                                        # (채널 256) 파라미터 512

        # 아래 층들은 학습되는 가중치가 없어서 하나만 만들어 여러 번 재사용 가능 (conv, bn은 층마다 따로 필요)
        self.pool    = nn.MaxPool2d(2, 2)          # (kernel_size=창문 크기 2×2, stride=이동 칸 수 2) 창문 안 최댓값만 남김 → 크기 절반, 채널 유지
        self.gap     = nn.AdaptiveAvgPool2d((1,1)) # (output_size=원하는 출력 크기 1×1) 창문 크기가 아님! 채널마다 14×14를 평균 내서 숫자 1개로
        self.dropout = nn.Dropout(p=0.3)           # (p=끌 확률) 학습 중에만 무작위로 30%를 0으로 → 과적합 방지. eval 모드에서는 꺼짐
        self.fc      = nn.Linear(256, num_classes) # (in_features=입력 개수 256, out_features=출력 개수 7) 파라미터 256×7 + 7 = 1,799

    # forward(self, x): 입력 x가 모델을 통과하는 순서. model(images)라고 쓰면 자동으로 호출됨
    # 한 줄은 안쪽부터 실행: ① conv(특징 추출) → ② bn(정규화) → ③ relu(음수 → 0) → ④ pool(크기 절반)
    # 주석의 B = 배치 크기
    def forward(self, x):                                     # x: (B, 3, 224, 224)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))   # (B, 32, 112, 112)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))   # (B, 64, 56, 56)
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))   # (B, 128, 28, 28)
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))   # (B, 256, 14, 14)
        x = self.gap(x)                                       # (B, 256, 1, 1)  채널마다 평균 1개
        x = x.view(x.size(0), -1)                             # (B, 256)  view(배치 크기, -1): 배치는 유지하고 나머지를 한 줄로 펼침. -1 = "나머지는 알아서 계산"
        x = self.dropout(x)                                   # (B, 256)  학습 중에만 30% 끔
        x = self.fc(x)                                        # (B, 7)    나이대 7개의 점수(logit). softmax는 CrossEntropyLoss가 내부에서 처리
        return x                                              # 7개 점수를 돌려줌

# SmallCNN(): 모델 생성 (이 순간 가중치가 무작위로 초기화됨 → 위에서 시드를 먼저 고정한 이유)
# .to(device): 모델의 모든 가중치를 GPU로 옮김
model = SmallCNN().to(device)    # 모델을 GPU로

# 확인용: 4-1에서 꺼낸 배치를 GPU로 옮겨 모델에 통과시켜 보기
out = model(images.to(device))   # (images.to(장치)) 데이터도 모델과 같은 장치에 있어야 계산됨
print(out.shape)                   # torch.Size([64, 7]) → 사진 64장 × 나이대 7개 점수

# model.parameters(): 학습되는 모든 가중치 텐서 / p.numel(): 텐서 안의 숫자 개수 / sum(...): 전부 더하기
n_params = sum(p.numel() for p in model.parameters())
print(n_params)                    # 390695 (손으로 계산한 설계값과 같아야 함)


# 프로그램 4-3 손실 함수와 최적화 알고리즘 설정하기
# =============================================================================
# 손실 함수: 모델이 얼마나 틀렸는지 숫자로 계산
# 최적화 알고리즘: 그 숫자가 줄어드는 방향으로 가중치를 바꾸는 방법
# =============================================================================

LR = 0.001                      # 학습률(learning rate): 한 번에 가중치를 움직이는 폭. 발표 자료 실험 조건(0.001 / 0.0001 / 0.00001) 중 하나

# nn.CrossEntropyLoss(): 다중 분류용 손실 함수
#   (출력 점수, 정답 번호)를 받아서 내부에서 softmax(점수 → 확률) 후 −log(정답 확률)을 계산
#   정답 확률이 높을수록 손실이 작음. 아무것도 모를 때(7개 균등) 약 1.95 = ln 7
#   ( ) 안에 weight=나이대별 가중치 를 넣으면 적은 나이대를 틀렸을 때 손실을 더 크게 줄 수 있음 (본 학습 때 적용 예정)
criterion = nn.CrossEntropyLoss()          # 다중 분류용 손실 함수

# torch.optim.Adam(학습할 가중치들, lr=학습률)
#   model.parameters(): 모델의 가중치 390,695개를 optimizer에 넘김 → 반드시 model.to(device) 다음에 만들어야 함
#   Adam: 가중치마다 이동 폭을 자동으로 조절하는 최적화 알고리즘
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(criterion)                # 확인용: CrossEntropyLoss()
print(optimizer)                # 확인용: Adam 설정 (lr: 0.001, betas: (0.9, 0.999), weight_decay: 0 등)

# 프로그램 4-4 학습 루프 실행하기
# =============================================================================
# 에폭마다: 학습(가중치 업데이트) → 검증(성능 확인) → 기록 → 최고 모델 저장
# 현재는 100장 과적합 테스트 설정 (증강 끔, 100 에폭)
# =============================================================================

# 과적합 테스트: train에도 증강을 끈다 (끝나면 이 줄을 지우거나 주석 처리)
# train_subset은 train_dataset을 가리키고 있어서, 원본의 transform만 바꾸면 부분집합에도 적용됨
train_dataset.transform = val_transform

NUM_EPOCHS = 100                    # 전체 데이터를 몇 번 반복해서 볼지. 과적합 테스트용 (본 학습은 팀원과 같은 20)
SAVE_PATH = "small_cnn_best.pth"    # 가장 좋은 모델의 가중치를 저장할 파일 이름 (프로젝트 폴더에 생성)

# 에폭마다 결과를 쌓아둘 딕셔너리. {이름표: 빈 리스트} 4개 → 4-5에서 학습 곡선 그릴 때 사용
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0                  # 지금까지 가장 높았던 검증 정확도 (처음엔 0)

# range(NUM_EPOCHS): 0, 1, ..., NUM_EPOCHS-1 → epoch 변수에 차례로 들어감
for epoch in range(NUM_EPOCHS):
    # ---------- 학습 ----------
    model.train()                   # 학습 모드: Dropout 켜짐, BN은 현재 배치 통계 사용. 매 에폭 시작마다 다시 호출해야 함 (검증 때 eval로 바뀌므로)
    running_loss, correct, total = 0.0, 0, 0   # 이번 에폭의 손실 합계, 맞힌 개수, 본 사진 수를 0으로 초기화

    for images, labels in train_loader:                       # 배치 하나씩 꺼냄: images (B, 3, 224, 224), labels (B,)
        images, labels = images.to(device), labels.to(device) # (장치) 데이터를 모델이 있는 GPU로 옮김

        optimizer.zero_grad()                      # 이전 기울기 초기화. PyTorch는 기울기를 누적하므로 매 배치 지워야 함
        outputs = model(images)                    # 순전파: (B, 7) 점수
        loss = criterion(outputs, labels)          # (모델 출력 점수, 정답 번호) → 배치 평균 손실
        loss.backward()                            # 역전파: 가중치마다 "바꾸면 손실이 얼마나 변하는지"(기울기) 계산해서 .grad에 저장
        optimizer.step()                           # 가중치 업데이트: 기울기를 보고 손실이 줄어드는 방향으로 lr만큼 이동

        # loss.item(): 텐서 → 파이썬 숫자 / images.size(0): 이번 배치의 사진 수 (0번 차원 = 배치)
        # 배치 평균 × 배치 크기 = 배치 합계 → 마지막 배치가 작아도(100 = 64 + 36) 정확한 전체 평균을 구할 수 있음
        running_loss += loss.item() * images.size(0)
        # outputs.argmax(dim=1): (B, 7)에서 1번 차원(나이대 7개) 중 가장 큰 점수의 위치 → 예측 나이대 번호 (B,)
        # (예측 == 정답) → True/False 배열 / .sum(): True 개수 / .item(): 파이썬 숫자로
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)                    # labels.size(0): 이번 배치의 사진 수

    train_loss = running_loss / total              # 에폭 전체 평균 손실
    train_acc = correct / total                    # 에폭 전체 정확도 (0~1)

    # ---------- 검증 ----------
    model.eval()                    # 평가 모드: Dropout 꺼짐, BN은 학습 중 누적한 통계 사용
    running_loss, correct, total = 0.0, 0, 0   # 검증용으로 다시 0으로 초기화

    # torch.no_grad(): 이 블록 안에서는 기울기를 기록하지 않음 → 메모리 절약, 속도 향상
    # 검증에서는 가중치를 바꾸면 안 되므로 zero_grad, backward, step이 없음
    with torch.no_grad():
        for images, labels in val_loader:                     # 검증 배치 하나씩
            images, labels = images.to(device), labels.to(device)   # 학습 루프와 같은 방식으로 GPU로
            outputs = model(images)                           # 순전파만
            loss = criterion(outputs, labels)                 # 손실 계산 (기록용)

            running_loss += loss.item() * images.size(0)      # 학습 구간과 같은 방식으로 합계
            correct += (outputs.argmax(dim=1) == labels).sum().item()   # 맞힌 개수
            total += labels.size(0)                           # 본 사진 수

    val_loss = running_loss / total                # 검증 평균 손실
    val_acc = correct / total                      # 검증 정확도

    # ---------- 기록 ----------
    # history["이름표"]: 해당 리스트를 꺼냄 / .append(값): 리스트 끝에 이번 에폭 값을 추가
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    # f"...{변수:형식}...": 변수 값을 문자열에 넣기
    #   {epoch+1:2d}: 정수를 2칸 폭으로 / {train_loss:.4f}: 소수점 4자리 / {train_acc:.3f}: 소수점 3자리
    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
          f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
          f"val loss {val_loss:.4f} acc {val_acc:.3f}")

    # ---------- 최고 모델 저장 ----------
    if val_acc > best_val_acc:                     # 이번 검증 정확도가 지금까지 최고보다 높으면
        best_val_acc = val_acc                     # 최고 기록 갱신
        # torch.save(저장할 것, 파일 경로)
        # model.state_dict(): 모델 전체가 아니라 가중치만 {층 이름: 값} 딕셔너리로 꺼냄
        #   → 나중에 SmallCNN()을 새로 만들고 load_state_dict로 불러오면 됨. 이전 파일은 덮어씀
        torch.save(model.state_dict(), SAVE_PATH)

print("best val acc:", best_val_acc)               # 학습이 모두 끝난 뒤 최고 검증 정확도 출력 (for문 밖)
