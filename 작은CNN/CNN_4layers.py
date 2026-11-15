# #############################################################################
#
#   작은 CNN(4층)으로 얼굴 사진의 나이대(7개) 분류하기
#   인공지능 4조 · 작은 CNN 담당: 임성빈(LIM)
#
#   [모듈 구성]
#     프로그램 4-1  데이터셋 구성          : 이미지 → 전처리·증강 → 텐서 배치
#     프로그램 4-2  모델 구성              : SmallCNN (Conv 블록 4개 + GAP + FC)
#     프로그램 4-3  손실 함수·최적화       : 가중치 CrossEntropyLoss + Adam
#     프로그램 4-4  학습 루프              : 학습 → 검증(acc·off1·macro-F1) → 기록 → 저장
#     프로그램 4-5  시각화                 : 예측 샘플 8장 + 학습 곡선
#
#   [읽는 법]
#     ▶ [모듈 시작] / ◀ [모듈 끝] 표시로 모듈 경계를 구분
#     각 모듈 머리말에 입력·출력을 적어 둠 → 모듈끼리 무엇을 주고받는지 알 수 있음
#     (값) 형식의 주석 = 함수 괄호 안 인자의 의미
#
#   [실행 방법]
#     프로젝트 폴더(whats_your_face_age)에서 .venv를 켠 뒤
#     python models\CNN_4layers_latest.py
#     (파일 이름을 CNN_4layers.py로 바꿨다면 python models\CNN_4layers.py)
#
#   [모드 전환]
#     USE_SUBSET = True  → 100장으로 빠른 점검 (증강 끔, 5 에폭, 1~2분)
#     USE_SUBSET = False → 본 학습 (학습 71,735장 / 검증 15,288장, 20 에폭, 약 1시간 50분)
#
#   [만들어지는 파일]
#     small_cnn_best.pth     : 검증 macro-F1이 가장 좋았던 에폭의 모델 가중치
#     small_cnn_history.csv  : 에폭별 학습 기록 (팀원 history.csv와 같은 열 이름)
#     pred_samples.png       : 예측 샘플 8장 그림
#     learning_curve.png     : 학습 곡선 그림
#     ※ 다시 실행하면 덮어쓰므로, 남길 결과는 results/ 폴더에 따로 백업할 것
#
# #############################################################################


# =============================================================================
#
#   프로그램 4-1  얼굴 나이대 데이터셋 구성하기
#
#   ▶ [모듈 4-1 시작]
#   입력: data/ 폴더의 얼굴 사진 파일 (training, validation)
#   출력: train_loader, val_loader (배치 단위 텐서), train_dataset, val_dataset, device
#
#   목적: data/ 폴더의 얼굴 이미지를 불러와서, 모델에 넣을 수 있는
#         (이미지 텐서, 나이대 번호) 묶음(배치)으로 만든다.
#
#   흐름: 설정값 → 전처리(transform) → 데이터셋(ImageFolder)
#         → 부분집합(Subset, 테스트용) → 배치 묶기(DataLoader) → 확인 출력
#
# =============================================================================

# ---------- 라이브러리 불러오기 ----------
import torch                                            # PyTorch 본체: 텐서 계산, GPU 사용, 난수 시드 고정 등
import os                                               # 운영체제 기능. 여기서는 폴더 경로를 합치는 os.path.join만 사용
from torchvision import datasets, transforms            # torchvision: 이미지용 도구 모음
                                                        #   datasets  : ImageFolder 등 데이터셋 불러오기 도구
                                                        #   transforms: 크기 조절·증강·텐서 변환·정규화 도구
from torch.utils.data import DataLoader, Subset         # DataLoader: 데이터셋을 배치 단위로 묶어서 꺼내주는 도구
                                                        # Subset    : 데이터셋 중 일부 번호의 사진만 고르는 도구

# ---------- 설정값 (상수: 실행 중 바뀌지 않는 값은 대문자 이름으로) ----------
DATA_DIR = "data"          # 데이터 폴더 이름. 프로젝트 폴더 기준 "상대 경로"
                           #   → 반드시 프로젝트 폴더에서 실행해야 data 폴더를 찾을 수 있음
                           #   → 절대 경로(C:\...)를 쓰지 않아서 팀원 PC(Mac 포함)에서도 그대로 동작
IMG_SIZE = 224             # 모델에 넣을 이미지 한 변의 크기 (224×224 픽셀). 팀원 모델과 동일
BATCH_SIZE = 64            # 한 번에 모델에 넣을 사진 수. 팀원 ResNet-50·ConvNeXt와 통일
SEED = 42                  # 난수 시드. 같은 시드면 매번 같은 "무작위" 결과가 나옴 (재현성). 팀원과 동일

USE_SUBSET = False      # 모드 스위치
                        #   True : 100장 과적합 테스트 (코드 점검용, 1~2분)
                        #   False: 본 학습 (전체 데이터, 약 1시간 50분)
torch.manual_seed(SEED)    # (시드값) PyTorch의 모든 무작위 동작(가중치 초기화, 셔플, 증강)을 고정.
                           # 모델을 만들기(4-2) "전에" 호출해야 초기 가중치까지 고정됨

# ImageNet 데이터셋의 채널별 평균·표준편차 (R, G, B 순서)
#   → 4-1 정규화(Normalize)와 4-5 시각화(정규화 되돌리기)에 같이 사용
#   → 값을 한 곳(상수)에서만 관리해야 두 곳의 값이 어긋나지 않음
MEAN = (0.485,0.456,0.406 )    # (R 평균, G 평균, B 평균)
STD = ( 0.229, 0.224, 0.225)   # (R 표준편차, G 표준편차, B 표준편차)

# ---------- 계산 장치 선택 (GPU / CPU) ----------
# 문법: (조건이 참일 때 값) if (조건) else (거짓일 때 값)
# torch.cuda.is_available(): NVIDIA GPU를 쓸 수 있으면 True → "cuda", 아니면 "cpu"
# torch.device(...)는 "어떤 장치를 쓸지" 이름표만 만든다.
#   실제로 GPU에서 계산하려면 모델과 데이터를 각각 .to(device)로 옮겨야 함 (4-2, 4-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU가 있으면 "cuda", 없으면 "cpu" 이름표를 device에 저장
print("using device:", device)                          # 확인용: using device: cuda 가 나와야 GPU 사용

# ---------- 학습용 transform: 확대·자르기 → 증강 → 텐서 → 정규화 ----------
# transforms.Compose([변환1, 변환2, ...]): 리스트 안의 변환을 위에서부터 순서대로 적용
# 팀원 코드(ResNet50 common.py, ConvNeXt train_convnext.py의 build_transforms)와 완전히 같은 설정
# 증강(무작위 변형)은 학습 데이터에만 넣는다 → 같은 사진도 에폭마다 조금씩 다르게 보여서 과적합 방지
train_transform = transforms.Compose([                      # 학습용 전처리 묶음 시작 (아래 6단계를 순서대로 적용)
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0), ratio=(0.95, 1.05)),  # (크기, scale=자를 면적 비율 범위, ratio=가로세로 비율 범위)
                                                                                   #   원본 면적의 90~100%를 거의 정사각형으로 잘라낸 뒤 224×224로 다시 늘림
                                                                                   #   → 얼굴이 살짝 확대되거나 위치가 조금 이동한 효과. 크기 맞추기도 같이 해줌
    transforms.RandomHorizontalFlip(p=0.5),             # (p=확률) 50% 확률로 좌우 반전. 얼굴은 좌우가 바뀌어도 나이는 같음
    transforms.RandomRotation(degrees=10),              # (degrees=최대 각도) -10° ~ +10° 사이에서 무작위 회전. 빈 모서리는 검은색으로 채워짐
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # (brightness=밝기 변화 폭, contrast=대비 변화 폭)
                                                           #   밝기·대비를 각각 0.8배 ~ 1.2배 사이에서 무작위로 바꿈 (조명 차이에 강해짐)
    transforms.ToTensor(),                              # PIL 이미지(H, W, 3채널, 0~255 정수) → 텐서(3, H, W, 0.0~1.0 실수). 채널이 맨 앞으로 옴
    transforms.Normalize(MEAN, STD),                    # (평균, 표준편차) 채널마다 (값 − 평균) / 표준편차 → 대략 평균 0, 표준편차 1로 맞춤
    ])                                                  # Compose 리스트 끝

# ---------- 검증용 transform: 크기 맞추기 → 텐서 → 정규화 (증강 없음) ----------
# 검증은 매번 같은 조건으로 채점해야 하므로 무작위 증강을 넣지 않는다
# 팀원 코드의 검증용 transform과 완전히 같은 설정
val_transform = transforms.Compose([                        # 검증용 전처리 묶음 시작 (아래 3단계, 증강 없음)
    transforms.Resize((IMG_SIZE, IMG_SIZE)),            # ((높이, 너비)) 괄호 두 겹 → 가로·세로를 모두 224로 맞춤
                                                        #   (참고: Resize(224)처럼 숫자 하나면 짧은 변만 224로, 비율 유지)
                                                        #   지금 데이터는 모두 224×224라서 결과는 같지만, 팀원과 코드까지 통일
    transforms.ToTensor(),                              # PIL 이미지 → 텐서 (0~1)
    transforms.Normalize(MEAN, STD),                    # (평균, 표준편차) 학습용과 같은 값으로 정규화
])                                                      # Compose 리스트 끝

# ---------- 데이터셋 만들기 ----------
# datasets.ImageFolder(root=폴더 경로, transform=전처리)
#   root 안의 "하위 폴더 이름"이 곧 라벨이 된다
#   폴더 이름을 문자열 순서로 정렬해서 0, 1, 2 ... 번호를 붙임
#     '0 9'→0, '10 19'→1, '20 29'→2, '30 39'→3, '40 49'→4, '50 59'→5, '60over'→6 (우연히 나이 순서와 같음)
#   transform: 사진을 하나 꺼낼 때마다 적용할 전처리
#   사진을 미리 다 읽어두지 않고, 꺼낼 때마다 읽어서 변환함 (메모리 절약, 증강이 매번 달라짐)
train_dataset = datasets.ImageFolder(                       # 학습 데이터셋 만들기 (data/training, 71,735장)
    root=os.path.join(DATA_DIR, "training"),            # os.path.join(기준 폴더, 하위 폴더) → "data\training". 학습 데이터 71,735장
                                                        #   ※ "C:\..." 처럼 \가 들어간 일반 문자열은 \t(탭) 등으로 해석돼 경로가 깨짐 → 상대 경로 사용
    transform=train_transform,                          # 학습용 transform (증강 포함)
)                                                       # ImageFolder 끝
val_dataset = datasets.ImageFolder(                         # 검증 데이터셋 만들기 (data/validation, 15,288장)
    root=os.path.join(DATA_DIR, "validation"),          # "data\validation". 검증 데이터 15,288장
                                                        #   ※ testing 폴더는 최종 평가 때 딱 한 번만 사용 (학습·검증에 절대 쓰지 않음)
    transform=val_transform,                            # 검증용 transform (증강 없음)
)                                                       # ImageFolder 끝

print(train_dataset.class_to_idx)          # 라벨 순서 확인: {'0 9': 0, '10 19': 1, ..., '60over': 6} (폴더 이름 → 번호)
print(len(train_dataset), len(val_dataset))  # len(데이터셋) = 사진 수. "71735 15288" 이어야 정상
                                             #   → 경로를 잘못 넣으면 에러 없이 다른 폴더를 읽을 수 있어서, 개수로 꼭 확인

# ---------- 소규모 부분집합 (USE_SUBSET = True 일 때 사용) ----------
N_TRAIN = 100      # 테스트용 학습 사진 수 (100장 과적합 테스트)
N_VAL   = 100      # 테스트용 검증 사진 수 (본 학습 때는 부분집합 없이 validation 전체를 사용)

# torch.Generator().manual_seed(시드): 부분집합 뽑기 전용 난수 생성기를 만들고 시드 고정
g = torch.Generator().manual_seed(SEED)    # 시드 고정 → 매번 같은 100장이 뽑힘
# torch.randperm(n, generator=g): 0 ~ n-1 을 무작위로 섞은 순서 (예: [5021, 13, 70002, ...])
#   [:N_TRAIN]: 그중 앞에서 N_TRAIN개만 / .tolist(): 텐서 → 파이썬 리스트
#   ImageFolder는 클래스(폴더) 순서로 정렬돼 있어서, range(100)처럼 앞에서 자르면 0~9세만 뽑힘 → 섞어서 뽑아야 함
train_idx = torch.randperm(len(train_dataset), generator=g)[:N_TRAIN].tolist()   # 학습 데이터에서 무작위 100개 번호
val_idx   = torch.randperm(len(val_dataset), generator=g)[:N_VAL].tolist()       # 검증 데이터에서 무작위 100개 번호

# Subset(원본 데이터셋, 번호 리스트): 원본 중 해당 번호의 사진만 쓰는 데이터셋 (사진을 복사하지 않고 번호만 기억)
train_subset = Subset(train_dataset,train_idx)          # 학습 100장
val_subset   = Subset(val_dataset,val_idx)              # 검증 100장

# ---------- DataLoader: 데이터셋을 배치로 묶기 ----------
# DataLoader(데이터셋, batch_size=배치 크기, shuffle=섞을지 여부)
#   for문으로 꺼낼 때마다 (이미지 배치, 라벨 배치)를 돌려줌
#   첫 번째 자리에는 "사진을 꺼낼 수 있는 데이터셋"이 들어가야 함 (N_TRAIN 같은 숫자는 안 됨)
#   shuffle=True : 에폭마다 순서를 섞음 (학습용)
#   shuffle=False: 항상 같은 순서 (검증용)

if USE_SUBSET:                                          # 테스트 모드
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)    # 학습 100장 (64장 + 36장, 에폭당 2배치)
    val_loader   = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)     # 검증 100장
else:                                                   # 본 학습 모드
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)   # 학습 71,735장 (에폭당 1,121배치)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)    # 검증 15,288장 (팀원과 동일)

# ---------- 확인용: 첫 배치 하나만 꺼내 보기 ----------
# iter(로더): 하나씩 꺼낼 수 있는 상태로 만들기 / next(...): 하나 꺼내기
images, labels = next(iter(train_loader))                   # 첫 배치 꺼내기 → images (64, 3, 224, 224), labels (64,)
print(images.shape)   # torch.Size([64, 3, 224, 224]) 이어야 정상 → (배치, 채널, 높이, 너비)
print(labels[:10])    # 앞에서 10개 라벨. 0~6 사이 숫자들이 섞여 나와야 정상
print(images.mean().item(), images.std().item())  # 배치 전체 평균·표준편차. 정규화됐으면 대략 0 근처, 1 근처
                                                  #   (.item(): 숫자 1개짜리 텐서 → 파이썬 숫자)


# -----------------------------------------------------------------------------
# ◀ [모듈 4-1 끝] 만들어진 것: device, train_dataset, val_dataset, train_loader, val_loader
# -----------------------------------------------------------------------------


# =============================================================================
#
#   프로그램 4-2  4층 CNN 모델 구성하기
#
#   ▶ [모듈 4-2 시작]
#   입력: 사진 배치 (B, 3, 224, 224)
#   출력: model (SmallCNN, GPU에 올라간 상태) → 사진마다 나이대 7개 점수 (B, 7)
#
#   구조: [Conv 3×3 → BatchNorm → ReLU → MaxPool] 블록 4개 → GAP → Dropout → Linear
#         블록마다 크기는 절반(224→112→56→28→14), 채널은 2배(3→32→64→128→256)
#         파라미터 수: 390,695개 (ResNet-50의 약 1/65)
#
#   설계 근거:
#     - 크기가 절반이 될 때 채널을 2배 → 층마다 계산량이 일정 (VGG·ResNet 관례)
#     - 시작 채널 32 → 처음부터 학습하는 기준 모델이라 과적합을 줄이려고 약 39만 개로 제한
#     - GAP → 그냥 펼치면 FC 입력이 256×14×14 = 50,176개라 너무 큼 → 256개로 줄임
#
# =============================================================================

import torch.nn as nn       # nn: 신경망 층(Conv2d, BatchNorm2d, Linear 등)이 들어 있는 모듈

NUM_CLASSES = 7             # 분류할 나이대 개수 (0~9, 10대, 20대, 30대, 40대, 50대, 60세 이상)

# nn.Module을 상속해서 모델을 정의
#   __init__: 사용할 층을 미리 만들어 둠 (학습되는 가중치가 여기서 생김)
#   forward : 데이터가 층을 지나가는 순서를 정함
class SmallCNN(nn.Module):                                  # 4층 CNN 모델 정의 (nn.Module을 상속 → PyTorch 모델의 기본 기능을 물려받음)
    def __init__(self, num_classes=NUM_CLASSES):        # (num_classes=출력 개수) 기본값 7
        super().__init__()                              # 부모 클래스(nn.Module)의 초기화. 층을 등록하려면 반드시 필요

        # nn.Conv2d(입력 채널, 출력 채널, kernel_size=필터 크기, padding=테두리 칸 수, bias=편향 사용 여부)
        #   출력 채널 = 필터 개수 = 찾을 패턴의 종류 수
        #   출력 크기 = (H + 2×padding − kernel_size) / stride + 1
        #             → (224 + 2×1 − 3) / 1 + 1 = 224 (padding=1이면 크기 유지)
        #             → padding=0이면 222 (K−1 = 2칸 감소)
        #   Conv"2d": 필터가 세로·가로 2방향으로 움직인다는 뜻 (채널은 한 번에 다 덮음)
        #   bias=False: 바로 뒤 BatchNorm이 평균을 빼므로 bias가 의미 없음
        #   conv 파라미터 수 = 입력 채널 × 출력 채널 × 3 × 3
        # nn.BatchNorm2d(채널 수): 채널마다 평균 0, 분산 1로 정규화 → 학습 안정화
        #   채널마다 γ(스케일), β(이동) 2개를 학습 → 파라미터 2 × 채널
        #   바로 앞 conv의 출력 채널 수와 같아야 함

        # 블록 1: 3 → 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)     # (RGB 3채널 입력, 필터 32개, 3×3, 테두리 1칸) 파라미터 3×32×3×3 = 864
        self.bn1   = nn.BatchNorm2d(32)                                         # (채널 32) 파라미터 2×32 = 64
        # 블록 2: 32 → 64
        self.conv2 = nn.Conv2d(32, 64,kernel_size=3, padding=1, bias=False)     # (32채널 입력, 필터 64개, 3×3, 테두리 1칸) 파라미터 32×64×9 = 18,432
        self.bn2   = nn.BatchNorm2d(64)                                         # (채널 64) 파라미터 2×64 = 128
        # 블록 3: 64 → 128
        self.conv3 = nn.Conv2d(64, 128,kernel_size=3, padding=1, bias=False)    # (64채널 입력, 필터 128개, 3×3, 테두리 1칸) 파라미터 64×128×9 = 73,728
        self.bn3   = nn.BatchNorm2d(128)                                        # (채널 128) 파라미터 2×128 = 256
        # 블록 4: 128 → 256
        self.conv4 = nn.Conv2d(128, 256,kernel_size=3, padding=1, bias=False)   # (128채널 입력, 필터 256개, 3×3, 테두리 1칸) 파라미터 128×256×9 = 294,912 (전체의 약 75%)
        self.bn4   = nn.BatchNorm2d(256)                                        # (채널 256) 파라미터 2×256 = 512

        # 아래 층들은 학습되는 가중치가 없어서 하나만 만들어 여러 번 재사용 가능 (conv, bn은 층마다 따로 필요)
        self.pool    = nn.MaxPool2d(2, 2)          # (kernel_size=창문 크기 2×2, stride=이동 칸 수 2)
                                                   #   창문 안 4칸 중 최댓값만 남김 → 크기 절반 ((224 − 2) / 2 + 1 = 112), 채널 유지
        self.gap     = nn.AdaptiveAvgPool2d((1,1)) # (output_size=원하는 출력 크기 1×1) ※ 창문 크기가 아님!
                                                   #   입력 크기와 상관없이 채널마다 전체(14×14)를 평균 내서 숫자 1개로 → (B, 256, 1, 1)
        self.dropout = nn.Dropout(p=0.3)           # (p=끌 확률) 학습 중에만 무작위로 30%를 0으로 → 특정 특징에만 의존하지 않게 해서 과적합 방지
                                                   #   model.eval() 상태에서는 꺼짐
        self.fc      = nn.Linear(256, num_classes) # (in_features=입력 개수 256, out_features=출력 개수 7)
                                                   #   모든 입력과 출력을 연결. 파라미터 256×7 + 7(bias) = 1,799

    # forward(self, x): 입력 x가 모델을 통과하는 순서. model(images)라고 쓰면 자동으로 호출됨
    # 한 줄은 "안쪽부터" 실행: ① conv(특징 추출, 채널 변경) → ② bn(정규화) → ③ relu(음수 → 0) → ④ pool(크기 절반)
    # 주석의 B = 배치 크기 (사진 장수)
    def forward(self, x):                                     # x: (B, 3, 224, 224)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))   # 블록 1 → (B, 32, 112, 112)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))   # 블록 2 → (B, 64, 56, 56)
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))   # 블록 3 → (B, 128, 28, 28)
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))   # 블록 4 → (B, 256, 14, 14)
        x = self.gap(x)                                       # GAP    → (B, 256, 1, 1)  채널마다 평균 1개
        x = x.view(x.size(0), -1)                             # 펼치기 → (B, 256)
                                                              #   view(배치 크기, -1): 배치는 유지하고 나머지를 한 줄로 펼침. -1 = "나머지는 알아서 계산"
                                                              #   (참고: view는 괄호에 "크기"를, permute는 "차원 번호"를 넣음)
        x = self.dropout(x)                                   # (B, 256)  학습 중에만 30% 끔
        x = self.fc(x)                                        # (B, 7)    나이대 7개의 점수(logit)
                                                              #   softmax(점수 → 확률)는 CrossEntropyLoss가 내부에서 처리하므로 여기서 하지 않음
        return x                                              # 7개 점수를 돌려줌

# SmallCNN(): 모델 생성 (이 순간 가중치가 무작위로 초기화됨 → 위에서 시드를 먼저 고정한 이유)
# .to(device): 모델의 모든 가중치를 GPU로 옮김
model = SmallCNN().to(device)    # 모델을 GPU로

# ---------- 확인용: 4-1에서 꺼낸 배치를 모델에 통과시켜 보기 ----------
out = model(images.to(device))   # (images.to(장치)) 데이터도 모델과 같은 장치(GPU)에 있어야 계산됨
print(out.shape)                 # torch.Size([64, 7]) → 사진 64장 × 나이대 7개 점수

# model.parameters(): 학습되는 모든 가중치 텐서들 / p.numel(): 텐서 안의 숫자 개수 / sum(...): 전부 더하기
n_params = sum(p.numel() for p in model.parameters())       # 학습되는 숫자(파라미터)의 총개수
print(n_params)                  # 390695 (손으로 계산한 설계값과 같아야 함)
                                 #   conv 864+18,432+73,728+294,912 + bn 64+128+256+512 + fc 1,799 = 390,695


# -----------------------------------------------------------------------------
# ◀ [모듈 4-2 끝] 만들어진 것: model (파라미터 390,695개, GPU)
# -----------------------------------------------------------------------------


# =============================================================================
#
#   프로그램 4-3  손실 함수와 최적화 알고리즘 설정하기
#
#   ▶ [모듈 4-3 시작]
#   입력: model의 파라미터, train_dataset의 라벨 목록
#   출력: criterion (가중 교차 엔트로피 손실), optimizer (Adam)
#
#   손실 함수       : 모델이 얼마나 틀렸는지 숫자로 계산 (작을수록 좋음)
#   최적화 알고리즘 : 그 숫자가 줄어드는 방향으로 가중치를 바꾸는 방법
#
# =============================================================================

LR = 0.001                      # 학습률(learning rate): 한 번에 가중치를 움직이는 폭
                                #   발표 자료 실험 조건(0.001 / 0.0001 / 0.00001) 중 하나. 교재 값과도 같음
                                #   처음부터 학습하는 모델이라 미세조정용(팀원 1e-4)보다 큰 값을 씀

# nn.CrossEntropyLoss(weight=나이대별 가중치): 다중 분류용 손실 함수
#   (출력 점수, 정답 번호)를 받아서 내부에서 softmax(점수 → 확률) 후 −log(정답 확률)을 계산
#   정답 확률이 높을수록 손실이 작음. 아무것도 모를 때(7개 균등) 약 1.95 = ln 7
#   weight=...: 적은 나이대를 틀렸을 때 손실을 더 크게 줌 (아래에서 계산해서 적용)

# torch.optim.Adam(학습할 가중치들, lr=학습률)
#   model.parameters(): 모델의 가중치 390,695개를 optimizer에 넘김 → 반드시 model.to(device) 다음에 만들어야 함
#   Adam: 가중치마다 이동 폭을 자동으로 조절하는 최적화 알고리즘
#         (betas=(0.9, 0.999), weight_decay=0 등은 기본값 사용)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)     # (학습할 가중치들, lr=학습률) Adam 최적화 알고리즘 생성

# ---------- 나이대별 손실 가중치 계산 (팀원 코드와 같은 식) ----------
# train_dataset.targets: ImageFolder가 미리 만들어 둔 "모든 학습 사진의 라벨 목록" (71,735개, 사진을 열지 않아 빠름)
# torch.tensor(...)    : 파이썬 리스트 → 텐서
# torch.bincount(라벨들, minlength=7): 0번이 몇 개, 1번이 몇 개, ... 6번이 몇 개인지 세기
#                                      minlength=7 → 0개인 번호가 있어도 7칸을 다 만듦
# .float()             : 다음 줄 나눗셈을 위해 정수 → 실수
counts = torch.bincount(torch.tensor(train_dataset.targets), minlength=NUM_CLASSES).float()   # [11561, 12000, 12000, 12000, 12104, 6801, 5269]
# 가중치 = 전체 사진 수 / (나이대 수 × 그 나이대 사진 수)
#   사진이 많은 나이대 → 1보다 작음 (20대 0.85) / 적은 나이대 → 1보다 큼 (50대 1.51, 60+ 1.94)
#   모든 나이대 사진 수가 같다면 전부 정확히 1.0
class_weights = counts.sum() / (NUM_CLASSES * counts)       # 나이대별 손실 가중치 = 전체 / (7 × 그 나이대 사진 수)

print("나이대별 사진 수:", counts.tolist())            # .tolist(): 텐서 → 파이썬 리스트 (보기 좋게 출력)
print("나이대별 손실 가중치:", class_weights)           # [0.8864, 0.8540, 0.8540, 0.8540, 0.8467, 1.5068, 1.9449]

# 가중치를 넣은 손실 함수. 가중치 텐서도 모델·데이터와 같은 장치(GPU)에 있어야 계산 가능 → .to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)) # (weight=나이대별 가중치) 가중 교차 엔트로피 손실 함수 생성


# -----------------------------------------------------------------------------
# ◀ [모듈 4-3 끝] 만들어진 것: optimizer (Adam, lr 0.001), criterion (나이대별 가중치 손실)
# -----------------------------------------------------------------------------


# =============================================================================
#
#   프로그램 4-4  학습 루프 실행하기
#
#   ▶ [모듈 4-4 시작]
#   입력: train_loader, val_loader, model, criterion, optimizer
#   출력: 학습된 model, small_cnn_best.pth (최고 모델), small_cnn_history.csv (기록), history
#
#   에폭마다: ① 학습(가중치 업데이트) → ② 검증(acc·off1·macro-F1)
#             → ③ 기록(history, CSV) → ④ 최고 모델 저장(기준: 검증 macro-F1)
#
#   USE_SUBSET = True : 100장, 증강 끔, 5 에폭 (코드 점검용)
#   USE_SUBSET = False: 전체 데이터, 증강 켬, 20 에폭 (본 학습)
#
# =============================================================================

import time                       # 에폭마다 걸린 시간 측정 (time.time(): 지금 시각을 초 단위 숫자로)
import pandas as pd               # 학습 기록을 표(DataFrame)로 만들어 CSV 파일로 저장

HISTORY_PATH = "small_cnn_history.csv"    # 학습 기록 CSV 파일 이름 (프로젝트 폴더에 생성, 매 에폭 덮어씀)


# 테스트 모드일 때만 학습 데이터의 증강을 끈다 (100장을 "외울 수 있는지"만 보는 점검이라 증강이 방해됨)
# train_subset은 train_dataset을 가리키고 있어서, 원본의 transform만 바꾸면 부분집합에도 적용됨
# 본 학습 모드에서는 이 줄이 실행되지 않으므로 4-1의 train_transform(증강 포함)이 그대로 쓰임
if USE_SUBSET:                                              # 점검 모드(True)일 때만 아래 줄 실행
    train_dataset.transform = val_transform   # 과적합 테스트일 때만 증강 끄기

from sklearn.metrics import f1_score     # macro-F1 계산용 (scikit-learn). 팀원 코드와 같은 함수
NUM_EPOCHS = 5 if USE_SUBSET else 20     # 전체 데이터를 몇 번 반복해서 볼지: 테스트일 때 5 / 본 학습일 때 20 (팀원과 동일)
SAVE_PATH = "small_cnn_best.pth"         # 가장 좋은 모델의 가중치를 저장할 파일 이름 (프로젝트 폴더에 생성)

# 에폭마다 결과를 쌓아둘 딕셔너리 {이름표: 빈 리스트}
#   → 매 에폭 CSV로 저장하고, 4-5에서 학습 곡선을 그릴 때 사용
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], # 에폭별 기록을 쌓을 {이름표: 빈 리스트} 7개
           "val_off1": [], "val_f1": [], "sec": []}     # 손실·정확도 + 한 칸 차이 정확도 + macro-F1 + 에폭 시간
best_val_f1 = 0.0                                       # 지금까지 가장 높았던 검증 macro-F1 (처음엔 0). 최고 모델 저장 기준

# range(NUM_EPOCHS): 0, 1, ..., NUM_EPOCHS-1 → epoch 변수에 차례로 들어감
for epoch in range(NUM_EPOCHS):                             # 에폭 반복: epoch = 0, 1, ..., NUM_EPOCHS−1 (본 학습 20번)
    # ---------------------------------------------------------------- ① 학습
    t0 = time.time()                # 에폭 시작 시각(초)
    model.train()                   # 학습 모드: Dropout 켜짐, BN은 현재 배치 통계 사용
                                    #   매 에폭 시작마다 다시 호출해야 함 (검증 때 eval로 바뀌므로)
    running_loss, correct, total = 0.0, 0, 0   # 이번 에폭의 손실 합계, 맞힌 개수, 본 사진 수를 0으로 초기화

    for images, labels in train_loader:                       # 배치 하나씩 꺼냄: images (B, 3, 224, 224), labels (B,)
        images, labels = images.to(device), labels.to(device) # (장치) 데이터를 모델이 있는 GPU로 옮김

        # 학습 5단계 (배치마다 반복)
        optimizer.zero_grad()                      # 1. 이전 기울기 초기화. PyTorch는 기울기를 "누적"하므로 매 배치 지워야 함
        outputs = model(images)                    # 2. 순전파: (B, 7) 점수
        loss = criterion(outputs, labels)          # 3. (모델 출력 점수, 정답 번호) → 배치 평균 손실 (나이대별 가중치 적용)
        loss.backward()                            # 4. 역전파: 가중치마다 "바꾸면 손실이 얼마나 변하는지"(기울기)를 계산해서 .grad에 저장
        optimizer.step()                           # 5. 가중치 업데이트: 기울기를 보고 손실이 줄어드는 방향으로 lr만큼 이동 (실제로 "배우는" 순간)

        # loss.item(): 텐서 → 파이썬 숫자 / images.size(0): 이번 배치의 사진 수 (0번 차원 = 배치)
        # 배치 평균 × 배치 크기 = 배치 합계 → 마지막 배치가 작아도 정확한 전체 평균을 구할 수 있음
        running_loss += loss.item() * images.size(0)        # 배치 손실 합계를 누적 (평균 × 사진 수)
        # outputs.argmax(dim=1): (B, 7)에서 1번 차원(나이대 7개) 중 가장 큰 점수의 위치 → 예측 나이대 번호 (B,)
        # (예측 == 정답) → True/False 배열 / .sum(): True 개수 / .item(): 파이썬 숫자로
        correct += (outputs.argmax(dim=1) == labels).sum().item() # 이번 배치에서 맞힌 개수를 누적
        total += labels.size(0)                    # labels.size(0): 이번 배치의 사진 수

    train_loss = running_loss / total              # 에폭 전체 평균 손실
    train_acc = correct / total                    # 에폭 전체 정확도 (0~1)
                                                   #   ※ 증강·Dropout이 켜진 상태에서, 에폭 동안 바뀌는 모델의 평균으로 잰 값
                                                   #     → 검증 정확도보다 낮게 나올 수 있음

    # ---------------------------------------------------------------- ② 검증
    model.eval()                                   # 평가 모드: Dropout 꺼짐, BN은 학습 중 누적한 통계 사용
    running_loss, correct, total = 0.0, 0, 0       # 검증용으로 다시 0으로 초기화
    all_preds, all_labels = [], []                 # 이번 에폭의 예측·정답을 모을 빈 리스트
                                                   #   F1은 배치별 값을 평균 내면 틀리므로, 전부 모았다가 마지막에 한 번에 계산

    # torch.no_grad(): 이 블록 안에서는 기울기를 기록하지 않음 → 메모리 절약, 속도 향상
    # 검증에서는 가중치를 바꾸면 안 되므로 zero_grad, backward, step이 없음
    #   (model.eval()은 "층의 동작 방식"을, no_grad()는 "기울기 기록"을 바꿈 → 역할이 달라 둘 다 필요)
    with torch.no_grad():                                   # 이 블록 안에서는 기울기를 기록하지 않음 (검증 전용)
        for images, labels in val_loader:                     # 검증 배치 하나씩
            images, labels = images.to(device), labels.to(device)   # 학습 루프와 같은 방식으로 GPU로
            outputs = model(images)                           # 순전파만
            loss = criterion(outputs, labels)                 # 손실 계산 (기록용, 팀원도 같은 가중치 손실로 기록)
            preds = outputs.argmax(dim=1)                     # 예측 나이대 번호 (한 번만 계산해서 아래에서 재사용)

            all_preds.append(preds.cpu())                     # .cpu(): sklearn은 GPU 텐서를 못 읽어서 CPU로 옮겨 모음
            all_labels.append(labels.cpu())                   # 정답도 같이 모음

            running_loss += loss.item() * images.size(0)      # 학습 구간과 같은 방식으로 손실 합계
            correct += (preds == labels).sum().item()         # 맞힌 개수
            total += labels.size(0)                           # 본 사진 수

    # torch.cat(텐서 리스트): 배치별 텐서들을 기존 차원으로 "이어붙이기" → [64개, 64개, ..., 56개] → 15,288개 하나
    #   (참고: torch.stack은 새 차원으로 "쌓기" → 4-5에서 사용)
    all_preds = torch.cat(all_preds)               # 전체 예측 (검증 사진 수,)
    all_labels = torch.cat(all_labels)             # 전체 정답 (검증 사진 수,)

    val_loss = running_loss / total                # 검증 평균 손실
    val_acc = correct / total                      # 검증 정확도: 예측 == 정답 비율
    # 한 칸 차이 정확도(off1): |예측 − 정답| ≤ 1 인 비율 → 바로 옆 나이대까지 정답으로 인정
    #   (all_preds - all_labels).abs(): 차이의 절댓값 (음수 방지: 20대로 예측, 정답 30대 → 2 − 3 = −1 → 1)
    #   <= 1: True/False / .float(): 1.0/0.0 / .mean(): 평균 = 비율 / .item(): 파이썬 숫자로
    #   항상 off1 ≥ acc 이어야 정상 (정확히 맞힌 사진은 한 칸 이내에도 포함되므로)
    val_off1 = ((all_preds - all_labels).abs() <= 1).float().mean().item() # 한 칸 차이 정확도 = |예측 − 정답| ≤ 1 인 비율
    # macro-F1: 나이대별 F1을 구해서 7개를 똑같이 1/7씩 평균 → 적은 나이대(50대·60+)도 같은 무게로 평가
    #   f1_score(정답, 예측, ...): .numpy()로 텐서 → numpy 배열 (sklearn 입력 형식)
    #   average="macro"   : 나이대별 F1의 단순 평균 ("micro"는 전체 한꺼번에, "weighted"는 사진 수로 가중)
    #   labels=[0, ..., 6]: 7개 나이대를 모두 계산에 포함 (어떤 나이대가 검증에 없어도 7개 기준으로 평균)
    #   zero_division=0   : 어떤 나이대를 한 번도 예측하지 않아 0으로 나누게 되면 에러 대신 0점 처리
    val_f1 = f1_score(all_labels.numpy(), all_preds.numpy(), # (정답 배열, 예측 배열, ...) macro-F1 계산
                      average="macro",                      # 나이대별 F1을 1/7씩 단순 평균
                      labels=list(range(NUM_CLASSES)), zero_division=0) # 0~6 일곱 나이대 모두 포함 / 0으로 나누게 되면 0점 처리

    # ---------------------------------------------------------------- ③ 기록
    # history["이름표"]: 해당 리스트를 꺼냄 / .append(값): 리스트 끝에 이번 에폭 값을 추가
    history["train_loss"].append(train_loss)                # 학습 평균 손실 기록
    history["train_acc"].append(train_acc)                  # 학습 정확도 기록
    history["val_loss"].append(val_loss)                    # 검증 평균 손실 기록
    history["val_acc"].append(val_acc)                      # 검증 정확도 기록
    history["val_off1"].append(val_off1)                    # 검증 한 칸 차이 정확도 기록
    history["val_f1"].append(val_f1)                        # 검증 macro-F1 기록

    sec = time.time() - t0                        # 이번 에폭에 걸린 시간(초) = 지금 시각 − 시작 시각
    history["sec"].append(sec)                    # ※ CSV 저장 "전에" 추가해야 함 (pandas 표는 모든 열의 길이가 같아야 함)

    # ---------- CSV 저장 (매 에폭) ----------
    # 매 에폭 저장하는 이유: 학습 중 에러가 나거나 창을 닫아도 그때까지의 기록이 남음 (팀원 코드와 같은 방식)
    df = pd.DataFrame(history)                                   # pd.DataFrame(딕셔너리): 이름표 → 열, 리스트 → 행 으로 된 표
    df.insert(0, "epoch", range(1, len(df) + 1))                 # (열 위치 0=맨 앞, 열 이름, 값들) 에폭 번호 1, 2, 3, ... 열 추가
    df = df.rename(columns={"val_off1": "val_off1_acc", "val_f1": "val_macro_f1"})   # (columns={원래 이름: 새 이름}) 팀원 history.csv와 같은 열 이름으로
    df.to_csv(HISTORY_PATH, index=False)                         # (파일 경로, index=False: 행 번호 0, 1, 2...는 따로 쓰지 않음)


    # 이번 에폭 결과 출력
    # f"...{변수:형식}...": 변수 값을 문자열에 넣기
    #   {epoch+1:2d}: 정수를 2칸 폭으로 / {train_loss:.4f}: 소수점 4자리 / {train_acc:.3f}: 소수점 3자리 / {sec:.0f}: 소수점 없이
    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "             # 에폭 번호 / 전체 에폭
          f"train loss {train_loss:.4f} acc {train_acc:.3f} | " # 학습 손실·정확도
          f"val loss {val_loss:.4f} acc {val_acc:.3f} off1 {val_off1:.3f} F1 {val_f1:.3f} | {sec:.0f}초") # 검증 손실·정확도·off1·F1 + 에폭 시간

    # ---------------------------------------------------------------- ④ 최고 모델 저장 (기준: 검증 macro-F1, 팀원과 동일)
    if val_f1 > best_val_f1:                       # 이번 에폭의 macro-F1이 지금까지 최고보다 높으면
        best_val_f1 = val_f1                       # 최고 기록 갱신
        # torch.save(저장할 것, 파일 경로)
        # model.state_dict(): 모델 전체가 아니라 가중치만 {층 이름: 값} 딕셔너리로 꺼냄
        #   → 나중에 SmallCNN()을 새로 만들고 load_state_dict로 불러오면 됨. 이전 파일은 덮어씀
        #   → 그래서 마지막 에폭이 아니라 "가장 좋았던 에폭"의 모델이 남음
        torch.save(model.state_dict(), SAVE_PATH)           # (가중치 딕셔너리, 파일 경로) 최고 모델 저장 (이전 파일 덮어씀)

print("best val macro-F1:", best_val_f1)           # 학습이 모두 끝난 뒤 최고 검증 macro-F1 출력 (for문 밖)


# -----------------------------------------------------------------------------
# ◀ [모듈 4-4 끝] 만들어진 것: small_cnn_best.pth, small_cnn_history.csv, history
# -----------------------------------------------------------------------------


# =============================================================================
#
#   프로그램 4-5  학습 모델 시각화하기
#
#   ▶ [모듈 4-5 시작]
#   입력: small_cnn_best.pth, val_dataset, history
#   출력: pred_samples.png, learning_curve.png
#
#   1부: 검증 사진 무작위 8장에 대한 예측 vs 정답 (맞으면 초록, 틀리면 빨강)
#   2부: 학습 곡선 (에폭별 loss, accuracy)
#
#   ※ plt.show()는 창을 닫을 때까지 멈춤 → 1부 창을 닫아야 2부 창이 뜸
#
# =============================================================================

import matplotlib.pyplot as plt      # 그래프·이미지를 그리는 라이브러리

CLASS_NAMES = ["0-9", "10s", "20s", "30s", "40s", "50s", "60+"]   # 라벨 번호 0~6 → 이름. 한글은 폰트 설정이 필요해서 영어로

# ---------------------------------------------------------------- 1부: 예측 샘플
# torch.load(파일): 저장된 가중치 딕셔너리 불러오기 / model.load_state_dict(...): 모델에 가중치 넣기
#   → 학습이 끝난 model은 "마지막 에폭" 상태이므로, 저장해둔 "최고 에폭" 가중치로 바꿔서 시각화
model.load_state_dict(torch.load(SAVE_PATH))                # 저장해 둔 최고 에폭 가중치를 불러와 모델에 넣기
model.eval()                                    # 평가 모드 (Dropout 끔)

# 검증 데이터 전체에서 고정 시드로 무작위 8장 고르기
#   (val_loader의 첫 배치를 쓰면 본 학습 모드에서는 폴더 순서상 전부 0~9세 사진이 나옴)
g_vis = torch.Generator().manual_seed(SEED)                               # 시각화 전용 난수 생성기 (매번 같은 8장)
vis_idx = torch.randperm(len(val_dataset), generator=g_vis)[:8].tolist()  # 검증 사진 번호 중 무작위 8개

# val_dataset[i]: (이미지 텐서, 라벨) 한 쌍 → [0]은 이미지, [1]은 라벨
# [... for i in vis_idx]: 8개 번호마다 꺼내서 리스트로
# torch.stack(텐서 리스트): 텐서들을 "새 차원(배치)으로 쌓기" → (3, 224, 224) 8개 → (8, 3, 224, 224)
sample_images = torch.stack([val_dataset[i][0] for i in vis_idx]) # 8장을 새 차원으로 쌓기 → (8, 3, 224, 224)
sample_labels = torch.tensor([val_dataset[i][1] for i in vis_idx])        # 라벨 8개 → 텐서 (8,)

with torch.no_grad():                                                     # 기울기 기록 안 함 (예측만)
    preds = model(sample_images.to(device)).argmax(dim=1).cpu()           # GPU로 보내 예측 → 가장 큰 점수 위치 → CPU로 되돌림 (matplotlib은 CPU 데이터만 그림)

plt.figure(figsize=(12, 6))                     # 새 그림 창 (figsize=(가로, 세로) 인치)
for i in range(8):                              # 8장 반복 (i = 0 ~ 7)
    plt.subplot(2, 4, i+1)                      # (행 수, 열 수, 칸 번호) 2행 4열 중 i+1번 칸. 칸 번호는 1부터 시작

    img = sample_images[i].permute(1, 2, 0)     # permute(새 순서에 올 "원래 차원 번호"): (3, H, W) → (H, W, 3)
                                                #   PyTorch는 (채널, 높이, 너비), matplotlib은 (높이, 너비, 채널) 순서를 원함
    img = img * torch.tensor(STD) + torch.tensor(MEAN)   # 정규화 되돌리기: x = x_norm × std + mean
                                                         #   permute로 채널을 맨 뒤로 보낸 "다음에" 계산해야 값 3개가 R, G, B에 하나씩 맞춰짐 (브로드캐스팅)
    img = img.clamp(0, 1)                       # (최솟값, 최댓값) 계산 오차로 0~1을 벗어난 값을 잘라냄
    plt.imshow(img)                             # 이미지 그리기

    pred, true = preds[i].item(), sample_labels[i].item()      # 예측 번호, 정답 번호 (파이썬 숫자로)
    color = "green" if pred == true else "red"                 # 맞으면 초록, 틀리면 빨강
    plt.title(f"pred: {CLASS_NAMES[pred]} / true: {CLASS_NAMES[true]}", color=color, fontsize=9)   # (제목 글자, color=글자색, fontsize=글자 크기)
    plt.axis('off')                             # 축 눈금 숨기기

plt.tight_layout()                              # 칸 사이 간격 자동 조절
plt.savefig("pred_samples.png", dpi=150)        # (파일 이름, dpi=해상도) 그림 저장. plt.show() 전에 해야 함
plt.show()                                      # 창 띄우기 (창을 닫아야 다음 줄로 진행)

# ---------------------------------------------------------------- 2부: 학습 곡선
# x축: 1, 2, ..., 에폭 수 (리스트 위치는 0부터지만 출력과 같게 1부터)
epochs = range(1, len(history["train_loss"]) + 1)           # x축 값: 1 ~ 에폭 수

plt.figure(figsize=(12, 4))                     # 새 그림 창

# 왼쪽 칸: loss
plt.subplot(1, 2, 1)                            # (1행, 2열, 1번 칸) ※ 반복문 밖이라 i+1이 아니라 숫자로 직접
plt.plot(epochs, history["train_loss"], label="train")    # (x값들, y값들, label=범례 이름) 학습 손실
plt.plot(epochs, history["val_loss"], label="val")        # 검증 손실
plt.xlabel("epoch")                             # x축 이름
plt.ylabel("loss")                              # y축 이름
plt.title("Loss")                               # 그래프 제목
plt.legend()                                    # 범례(train, val) 표시

# 오른쪽 칸: 정확도
plt.subplot(1, 2, 2)                            # (1행, 2열, 2번 칸)
plt.plot(epochs, history["train_acc"], label="train")     # 학습 정확도
plt.plot(epochs, history["val_acc"], label="val")         # 검증 정확도
plt.xlabel("epoch")                             # x축 이름
plt.ylabel("accuracy")                          # y축 이름
plt.title("Accuracy")                           # 그래프 제목
plt.legend()                                    # 범례 표시

plt.tight_layout()                              # 간격 자동 조절
plt.savefig("learning_curve.png", dpi=150)      # 그림 저장
plt.show()                                      # 창 띄우기

# -----------------------------------------------------------------------------
# ◀ [모듈 4-5 끝] 만들어진 것: pred_samples.png, learning_curve.png
# -----------------------------------------------------------------------------
