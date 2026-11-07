#프로그램 4-1 얼굴 나이대 데이터 셋 구성하기

import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

DATA_DIR = "data"
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42

MEAN = (0.485,0.456,0.406 )
STD = ( 0.229, 0.224, 0.225)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),          # 짧은 변을 224로 (비율 유지)
    transforms.CenterCrop(IMG_SIZE),      # 가운데 224×224
    transforms.RandomHorizontalFlip(p=0.5),                # 좌우 반전
    transforms.RandomRotation(degrees=10),           # ±10도 회전
    transforms.ColorJitter(brightness=0.2),       # 밝기 변화
    transforms.ToTensor(),                     # PIL 이미지 → 텐서 (0~1)
    transforms.Normalize(MEAN, STD),
    ])

# 검증용: 증강 없이 크기 맞추기 + 텐서 + 정규화만
val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE), 
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ---------- dataset ----------
train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "training"),
    transform=train_transform,
)
val_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "validation"),
    transform=val_transform,
)

print(train_dataset.class_to_idx)          # 라벨 순서 확인
print(len(train_dataset), len(val_dataset))

# ---------- 소규모 부분집합 ----------
N_TRAIN = 100      # 1단계: 100장 과적합 테스트 → 나중에 늘리기
N_VAL   = 100

g = torch.Generator().manual_seed(SEED)    # 시드 고정 → 매번 같은 부분집합
train_idx = torch.randperm(len(train_dataset), generator=g)[:N_TRAIN].tolist()
val_idx   = torch.randperm(len(val_dataset), generator=g)[:N_VAL].tolist()

train_subset = Subset(train_dataset,train_idx)
val_subset   = Subset(val_dataset,val_idx)

# ---------- dataloader ----------
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False)

images, labels = next(iter(train_loader))
print(images.shape)   # torch.Size([32, 3, 224, 224]) 이어야 정상
print(labels[:10])    # 0~6 사이 숫자들이 섞여 나와야 정상
print(images.mean().item(), images.std().item())  # 정규화됐으면 대략 0 근처, 1 근처

