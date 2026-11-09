"""여러 스크립트가 같이 쓰는 설정과 함수 모음.
나이대 정의, 모델 만들기, 사진 변환(전처리·증강), 데이터셋, 평가 지표가 들어 있다.
"""
import numpy as np
import pandas as pd
from PIL import Image

# 7개 나이대 (숫자 0~6이 이 순서와 같다)
BANDS = ["0-9세", "10대", "20대", "30대", "40대", "50대", "60세 이상"]
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
def build_transforms(train: bool, augment: bool = True):
    """train=True이고 augment=True일 때만 증강을 넣는다. 검증·시험은 정규화만."""
    from torchvision import transforms as T
    steps = []
    if train and augment:
        steps += [
            T.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
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
    import torch.nn as nn
    if name == "small_cnn":
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        class SmallCNN(nn.Module):
            """사전학습 없이 처음부터 학습하는 비교 기준 모델 (합성곱 블록 4개)."""

            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, NUM_CLASSES))

            def forward(self, x):
                return self.head(self.pool(self.features(x)))

            def get_classifier(self):
                return self.head

        return SmallCNN()

    import timm
    return timm.create_model(name, pretrained=pretrained, num_classes=NUM_CLASSES)


def gradcam_target_layer(model, name: str):
    """Grad-CAM이 들여다볼 마지막 합성곱 층."""
    if name == "small_cnn":
        return model.features[-1]
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
