"""ConvNeXt-Tiny 나이대 분류 학습 (한 파일 버전)

step4_train.py --model convnext_tiny 와 같은 설정을 한 파일에 모은 것이다.
다른 코드 파일 없이 이 파일 하나로 학습할 수 있다.

실행 (C:\\age_project 에서):
    python train_convnext.py
옵션 예:
    python train_convnext.py --epochs 20 --batch-size 64 --resume

결과: results/convnext_tiny/
    best.pt      검증 macro-F1이 가장 좋았던 모델
    last.pt      이어하기(--resume)용
    history.csv  에폭별 손실·점수 (학습 곡선용)
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# ------------------------------------------------------------------ 기본 설정
BANDS = ["0-9세", "10대", "20대", "30대", "40대", "50대", "60세 이상"]
NUM_CLASSES = 7
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)   # 정규화: 사전학습 때 쓴 ImageNet 기준
IMAGENET_STD = (0.229, 0.224, 0.225)


# ------------------------------------------------------------------ 데이터
def build_transforms(train: bool):
    if train:  # 증강은 학습 데이터에만
        steps = [
            T.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0), ratio=(0.95, 1.05)),  # 살짝 확대·자르기
            T.RandomHorizontalFlip(),                                             # 좌우 반전
            T.RandomRotation(10),                                                 # ±10도 회전
            T.ColorJitter(brightness=0.2, contrast=0.2),                          # 밝기·대비 ±20%
        ]
    else:
        steps = [T.Resize((IMG_SIZE, IMG_SIZE))]
    steps += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)


class FaceDataset(Dataset):
    """CSV(path, band)를 읽어 (사진, 나이대 번호)를 돌려준다."""

    def __init__(self, csv_path, image_root, transform):
        df = pd.read_csv(csv_path)
        self.paths = [os.path.join(image_root, p) for p in df["path"]]
        self.labels = df["band"].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


# ------------------------------------------------------------------ 평가 지표
def compute_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {
        "acc": float((y == p).mean()),                       # 정확도
        "off1_acc": float((np.abs(y - p) <= 1).mean()),      # 한 칸 차이 정확도
        "macro_f1": float(f1_score(y, p, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)),
    }


# ------------------------------------------------------------------ 한 에폭 학습/검증
def run_epoch(model, loader, device, criterion, optimizer=None, scheduler=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total, ys, ps = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            out = model(x)
            loss = criterion(out, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        total += loss.item() * len(y)
        ys.append(y.cpu().numpy())
        ps.append(out.argmax(1).cpu().numpy())
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    return total / len(ys), compute_metrics(ys, ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="splits_csv/train.csv")
    ap.add_argument("--val-csv", default="splits_csv/val.csv")
    ap.add_argument("--image-root", default="split_result_224")
    ap.add_argument("--out-dir", default="results/convnext_tiny")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)        # 몸통 학습률, 출력층은 10배
    ap.add_argument("--freeze-epochs", type=int, default=1)  # 처음 몇 에폭은 출력층만 학습
    ap.add_argument("--patience", type=int, default=4)       # 조기 종료
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out_dir, exist_ok=True)

    # ---- 데이터
    tr_ds = FaceDataset(a.train_csv, a.image_root, build_transforms(train=True))
    va_ds = FaceDataset(a.val_csv, a.image_root, build_transforms(train=False))
    tr_dl = DataLoader(tr_ds, a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True)
    va_dl = DataLoader(va_ds, a.batch_size * 2, shuffle=False, num_workers=a.workers, pin_memory=True)
    print(f"장치: {device} | 모델: convnext_tiny | 학습 {len(tr_ds):,}장 / 검증 {len(va_ds):,}장")

    # ---- 모델: ImageNet 사전학습 ConvNeXt-Tiny, 출력층 7칸
    model = timm.create_model("convnext_tiny", pretrained=True, num_classes=NUM_CLASSES).to(device)
    head_params = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head_params}
    body_params = [p for p in model.parameters() if id(p) not in head_ids]

    # ---- 손실: 사진이 적은 나이대일수록 큰 가중치
    counts = np.bincount(tr_ds.labels, minlength=NUM_CLASSES).astype(float)
    counts[counts == 0] = 1
    weight = torch.tensor(counts.sum() / (NUM_CLASSES * counts), dtype=torch.float32, device=device)
    print("나이대별 손실 가중치:", np.round(weight.cpu().numpy(), 2))
    criterion = nn.CrossEntropyLoss(weight=weight)

    # ---- 최적화와 학습률 스케줄 (1에폭 워밍업 → 코사인 감소)
    optimizer = torch.optim.AdamW([{"params": body_params, "lr": a.lr},
                                   {"params": head_params, "lr": a.lr * 10}], weight_decay=0.05)
    steps_per_epoch = max(1, len(tr_dl))
    total_steps, warm = a.epochs * steps_per_epoch, steps_per_epoch

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total_steps - warm)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # ---- 이어하기
    start, best_f1, bad, history = 0, -1.0, 0, []
    last_path, best_path = os.path.join(a.out_dir, "last.pt"), os.path.join(a.out_dir, "best.pt")
    if a.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start, best_f1, bad, history = ck["epoch"] + 1, ck["best_f1"], ck["bad"], ck["history"]
        print(f"{start}에폭부터 이어서 학습")

    # ---- 학습 반복
    for epoch in range(start, a.epochs):
        head_only = epoch < a.freeze_epochs        # 2단계 학습: 처음엔 출력층만
        for p in body_params:
            p.requires_grad = not head_only

        t0 = time.time()
        tr_loss, tr_m = run_epoch(model, tr_dl, device, criterion, optimizer, scheduler, scaler)
        va_loss, va_m = run_epoch(model, va_dl, device, criterion)
        row = {"epoch": epoch + 1, "head_only": head_only, "train_loss": tr_loss, "train_acc": tr_m["acc"],
               "val_loss": va_loss, "val_acc": va_m["acc"], "val_off1_acc": va_m["off1_acc"],
               "val_macro_f1": va_m["macro_f1"], "sec": time.time() - t0}
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(a.out_dir, "history.csv"), index=False)
        print(f"[{epoch + 1:2d}/{a.epochs}] 학습 손실 {tr_loss:.3f} 정확도 {tr_m['acc']:.3f} | "
              f"검증 손실 {va_loss:.3f} 정확도 {va_m['acc']:.3f} 한칸 {va_m['off1_acc']:.3f} "
              f"F1 {va_m['macro_f1']:.3f} | {row['sec']:.0f}초")

        if va_m["macro_f1"] > best_f1:             # 가장 좋은 모델 저장
            best_f1, bad = va_m["macro_f1"], 0
            torch.save({"model": model.state_dict(), "model_name": "convnext_tiny", "epoch": epoch + 1}, best_path)
            print("  → 가장 좋은 모델 저장")
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                    "best_f1": best_f1, "bad": bad, "history": history}, last_path)
        if bad >= a.patience:                      # 조기 종료
            print(f"검증 F1이 {a.patience}번 연속 오르지 않아 조기 종료")
            break

    print(f"끝. 최고 검증 macro-F1 {best_f1:.3f} → {best_path}")


if __name__ == "__main__":
    main()
