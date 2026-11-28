"""여러 스크립트가 같이 쓰는 설정과 함수 모음.
나이대 정의, 모델 만들기, 사진 변환(전처리·증강), 데이터셋, 평가 지표가 들어 있다.
"""
import os

import numpy as np
import pandas as pd
from PIL import Image

# 분류할 클래스. 기본은 7개 나이대(숫자 0~6이 이 순서와 같다).
# cmd에서 'set AGE_GROUPS=3'을 먼저 실행하면 3그룹(소아청소년 / 성인 / 노인)으로 바뀐다.
AGE_GROUPS = int(os.environ.get("AGE_GROUPS", "7"))
BANDS_7 = ["0-9세", "10대", "20대", "30대", "40대", "50대", "60세 이상"]
GROUPS_3 = ["소아청소년(0~10대)", "성인(20~50대)", "노인(60세 이상)"]
GROUP_OF_BAND = [0, 0, 1, 1, 1, 1, 2]  # 7개 나이대 번호 → 3그룹 번호
if AGE_GROUPS == 3:
    BANDS = GROUPS_3
elif AGE_GROUPS == 7:
    BANDS = BANDS_7
else:
    raise SystemExit(f"AGE_GROUPS는 7 또는 3이어야 합니다 (지금: {AGE_GROUPS})")
NUM_CLASSES = len(BANDS)

# 사전학습 모델이 학습할 때 쓴 ImageNet 평균·표준편차 (정규화에 사용)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224


def age_to_band(age: int) -> int:
    """정확한 나이(숫자)를 나이대 번호(0~6)로 바꾼다."""
    age = int(age)
    if age < 0:
        raise ValueError(age)
    return min(age // 10, 6)


# ---------------------------------------------------------------- 사진 변환
# ---------------------------------------------------------------- 다른 얼굴 붙이기
def paste_face(img, other, rng, size_frac=(0.30, 0.45), center_frac=0.70):
    """img(메인 얼굴 사진) 구석에 other(다른 사람 사진)의 얼굴 부분을 작게 붙인다.
    옆 사람 얼굴이 사진 가장자리에 걸친 상황을 흉내 낸다. 일부는 사진 밖으로 잘린다.
    rng: random.Random (같은 rng면 항상 같은 결과 → 평가용 사진을 똑같이 만들 수 있음)"""
    img = img.copy()
    W, H = img.size
    ow, oh = other.size
    c = center_frac  # 다른 사진의 가운데(얼굴이 있는 부분)만 사용
    face = other.crop((int(ow * (1 - c) / 2), int(oh * (1 - c) / 2), int(ow * (1 + c) / 2), int(oh * (1 + c) / 2)))
    s = int(W * rng.uniform(*size_frac))
    face = face.resize((s, s), Image.BICUBIC)
    out = int(s * rng.uniform(0.0, 0.3))  # 사진 밖으로 나가는 정도 (0~30%)
    corner = rng.choice(["lt", "rt", "lb", "rb"])
    x = -out if corner[0] == "l" else W - s + out
    y = -out if corner[1] == "t" else H - s + out
    img.paste(face, (x, y))
    return img


class RandomPasteFace:
    """학습용 증강: 확률 p로 학습 사진 중 다른 한 장의 얼굴을 구석에 작게 붙인다. 정답은 메인 얼굴 그대로."""

    def __init__(self, paths, p=0.3):
        self.paths, self.p = list(paths), p

    def __call__(self, img):
        import random
        if random.random() >= self.p:
            return img
        other = Image.open(random.choice(self.paths)).convert("RGB")
        return paste_face(img, other, random)


def build_transforms(train: bool, augment: bool = True, blur: bool = False, paste_paths=None, paste_p: float = 0.0):
    """train=True이고 augment=True일 때만 증강을 넣는다. 검증·시험은 정규화만.
    blur=True이면 학습 증강에 '흐림 증강'(초점 흐림, 저해상도)을 추가한다.
    paste_p>0이면 학습 증강에 '얼굴 붙이기 증강'(다른 사람 얼굴을 구석에 작게 붙임)을 추가한다."""
    from torchvision import transforms as T
    steps = []
    if train and augment:
        if paste_p > 0 and paste_paths:
            steps.append(RandomPasteFace(paste_paths, paste_p))
        steps += [
            T.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        if blur:
            steps += [
                # 초점이 안 맞은 사진 흉내: 20% 확률로 가우시안 블러 (흐림 정도 무작위)
                T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.5, 2.5))], p=0.2),
                # 작은 사진을 확대한 것 흉내: 20% 확률로 작게 줄였다가 다시 224로 키움
                T.RandomApply([T.RandomChoice([
                    T.Compose([T.Resize(56), T.Resize((IMG_SIZE, IMG_SIZE))]),
                    T.Compose([T.Resize(80), T.Resize((IMG_SIZE, IMG_SIZE))]),
                    T.Compose([T.Resize(112), T.Resize((IMG_SIZE, IMG_SIZE))]),
                ])], p=0.2),
            ]
    else:
        steps += [T.Resize((IMG_SIZE, IMG_SIZE))]
    steps += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)


class FaceDataset:
    """split CSV(path, band, ...)를 읽어 (사진 텐서, 나이대 번호)를 돌려준다."""

    def __init__(self, df: pd.DataFrame, image_root: str, transform):
        import os
        self.paths = [os.path.join(image_root, p) for p in df["path"]]
        self.labels = df["band"].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


# ---------------------------------------------------------------- 모델
def build_model(name: str, pretrained: bool = True):
    """name: small_cnn / resnet50 / convnext_tiny (timm에 있는 다른 이름도 가능)."""
    import torch
    import torch.nn as nn
    if name == "small_cnn":
        # 팀원(LIM) 작은CNN/CNN_4layers.py의 SmallCNN과 같은 구조 (파라미터 390,695개)
        # [Conv 3x3 → BatchNorm → ReLU → MaxPool] 블록 4개 → GAP → Dropout(0.3) → Linear
        class SmallCNN(nn.Module):
            """사전학습 없이 처음부터 학습하는 비교 기준 모델 (합성곱 4층)."""

            def __init__(self, num_classes=NUM_CLASSES):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(32)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(64)
                self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
                self.bn3 = nn.BatchNorm2d(128)
                self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
                self.bn4 = nn.BatchNorm2d(256)
                self.pool = nn.MaxPool2d(2, 2)
                self.gap = nn.AdaptiveAvgPool2d((1, 1))
                self.dropout = nn.Dropout(p=0.3)
                self.fc = nn.Linear(256, num_classes)

            def forward(self, x):
                x = self.pool(torch.relu(self.bn1(self.conv1(x))))
                x = self.pool(torch.relu(self.bn2(self.conv2(x))))
                x = self.pool(torch.relu(self.bn3(self.conv3(x))))
                x = self.pool(torch.relu(self.bn4(self.conv4(x))))
                x = self.gap(x).flatten(1)
                return self.fc(self.dropout(x))

            def get_classifier(self):
                return self.fc

        return SmallCNN()

    import timm
    return timm.create_model(name, pretrained=pretrained, num_classes=NUM_CLASSES)


def gradcam_target_layer(model, name: str):
    """Grad-CAM이 들여다볼 마지막 합성곱 층."""
    if name == "small_cnn":
        return model.conv4
    if name.startswith("resnet"):
        return model.layer4[-1]
    if name.startswith("convnext"):
        return model.stages[-1]
    raise ValueError(f"{name}의 Grad-CAM 층을 common.py에 추가하세요")


# ---------------------------------------------------------------- 평가 지표
def compute_metrics(y_true, y_pred) -> dict:
    """정확도, 한 칸 차이 정확도, macro-F1."""
    from sklearn.metrics import f1_score
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "acc": float((y_true == y_pred).mean()),
        "off1_acc": float((np.abs(y_true - y_pred) <= 1).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)),
    }
