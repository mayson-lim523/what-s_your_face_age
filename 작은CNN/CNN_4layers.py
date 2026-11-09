#프로그램 4-1 얼굴 나이대 데이터 셋 구성하기

import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

DATA_DIR = "data"
IMG_SIZE = 224
BATCH_SIZE = 64
SEED = 42

torch.manual_seed(SEED)

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

# 프로그램 4-2 4층 CNN 모델 구성하기

import torch.nn as nn

NUM_CLASSES = 7

class SmallCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # 블록 1: 3 → 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32) ### ()안은 채널수...?
        # 블록 2: 32 → 64
        self.conv2 = nn.Conv2d(32, 64,kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        # 블록 3: 64 → 128
        self.conv3 = nn.Conv2d(64, 128,kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(128)
        # 블록 4: 128 → 256
        self.conv4 = nn.Conv2d(128, 256,kernel_size=3, padding=1, bias=False)
        self.bn4   = nn.BatchNorm2d(256)

        self.pool    = nn.MaxPool2d(2, 2)   ###(kernel_size, stride)
        self.gap     = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.3)
        self.fc      = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))   # (B, 32, 112, 112)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))   # (B, 64, 56, 56)
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))   # (B, 128, 28, 28)
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))   # (B, 256, 14, 14)
        x = self.gap(x)                                       # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)                             # (B, 256)
        x = self.dropout(x)
        x = self.fc(x)                                        # (B, 7)
        return x

model = SmallCNN().to(device)    # 모델을 GPU로

out = model(images.to(device))
print(out.shape)                   # torch.Size([32, 7])

n_params = sum(p.numel() for p in model.parameters())
print(n_params)                    # 390695


# 프로그램 4-3 손실 함수와 최적화 알고리즘 설정하기

LR = 0.001                      # 발표 자료 실험 조건 중 하나

criterion = nn.CrossEntropyLoss()          # 다중 분류용 손실 함수
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(criterion)
print(optimizer)

# 프로그램 4-4 학습 루프 실행하기

# 과적합 테스트: train에도 증강을 끈다 (끝나면 이 줄을 지우거나 주석 처리)
train_dataset.transform = val_transform

NUM_EPOCHS = 100
SAVE_PATH = "small_cnn_best.pth"

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    # ---------- 학습 ----------
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()                      # 이전 기울기 초기화
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()                           # 역전파
        optimizer.step()                      # 가중치 업데이트

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # ---------- 검증 ----------
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)                              # 학습 루프와 같은 방식으로 GPU로
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total

    # ---------- 기록 ----------
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)    
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)                                  # 나머지 3개도 append

    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
          f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
          f"val loss {val_loss:.4f} acc {val_acc:.3f}")

    # ---------- 최고 모델 저장 ----------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)

print("best val acc:", best_val_acc)