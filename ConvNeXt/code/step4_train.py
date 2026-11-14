"""4단계: 모델 학습. 비교 실험은 이 파일의 옵션만 바꿔서 실행한다.

사용 예 (기준 실험):
python step4_train.py --name base_resnet50 --model resnet50 \
  --train-csv splits/train_small.csv --val-csv splits/val_small.csv \
  --image-root processed --out-dir results

주요 옵션
  --model        small_cnn / resnet50 / convnext_tiny
  --train-mode   full(전체 미세조정, 기본) / head(마지막 층만 학습)
  --lr           몸통 학습률 (기본 0.0001). 새 출력층은 이 값의 10배
  --no-augment   증강 끄기
  --no-class-weight  나이대별 손실 가중치 끄기
  --epochs 20 --patience 4 --batch-size 64
  --resume       끊긴 학습 이어하기 (results/<name>/last.pt 사용)
  --exclude-source aaf,afad  보강 데이터 빼고 학습 (실험 3)

결과: results/<name>/best.pt(가장 좋은 모델), last.pt(이어하기용), history.csv(학습 곡선용), config.json
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import NUM_CLASSES, FaceDataset, build_model, build_transforms, compute_metrics


def set_trainable(model, head_only: bool):
    head = model.get_classifier()
    head_ids = {id(p) for p in head.parameters()}
    for p in model.parameters():
        p.requires_grad = (not head_only) or (id(p) in head_ids)


def run_epoch(model, loader, device, criterion, optimizer=None, scheduler=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total_loss, ys, ps = 0.0, [], []
    use_amp = device == "cuda"
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(device_type="cuda", enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        total_loss += loss.item() * len(y)
        ys.append(y.cpu().numpy())
        ps.append(out.argmax(1).cpu().numpy())
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    return total_loss / len(ys), compute_metrics(ys, ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="실험 이름 (결과 폴더 이름)")
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--train-mode", choices=["full", "head"], default="full")
    ap.add_argument("--freeze-epochs", type=int, default=1, help="full 모드에서 처음 몇 에폭은 출력층만 학습")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--no-class-weight", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--exclude-source", default="", help="예: aaf,afad")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(a.out_dir, a.name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(a), f, ensure_ascii=False, indent=2)
    print(f"장치: {device} | 실험: {a.name}")

    tr_df, va_df = pd.read_csv(a.train_csv), pd.read_csv(a.val_csv)
    if a.exclude_source:
        drop = a.exclude_source.split(",")
        tr_df = tr_df[~tr_df["source"].isin(drop)]  # 학습에서만 빼고, 검증은 그대로 두어 공정하게 비교
    print(f"학습 {len(tr_df):,}장 / 검증 {len(va_df):,}장")

    tr_ds = FaceDataset(tr_df, a.image_root, build_transforms(train=True, augment=not a.no_augment))
    va_ds = FaceDataset(va_df, a.image_root, build_transforms(train=False))
    tr_dl = DataLoader(tr_ds, a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True)
    va_dl = DataLoader(va_ds, a.batch_size * 2, shuffle=False, num_workers=a.workers, pin_memory=True)

    model = build_model(a.model, pretrained=not a.no_pretrained).to(device)

    # 나이대별 손실 가중치: 사진이 적은 나이대일수록 크게 (전체/(나이대 수 x 그 나이대 장수))
    weight = None
    if not a.no_class_weight:
        counts = np.bincount(tr_df["band"].astype(int), minlength=NUM_CLASSES).astype(float)
        counts[counts == 0] = 1
        weight = torch.tensor(counts.sum() / (NUM_CLASSES * counts), dtype=torch.float32, device=device)
        print("나이대별 손실 가중치:", np.round(weight.cpu().numpy(), 2))
    criterion = nn.CrossEntropyLoss(weight=weight)

    head_ids = {id(p) for p in model.get_classifier().parameters()}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    head = [p for p in model.parameters() if id(p) in head_ids]
    head_lr = a.lr * 10 if a.model != "small_cnn" else a.lr
    optimizer = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": head, "lr": head_lr}], weight_decay=0.05)

    # 학습률 스케줄: 첫 1에폭 동안 서서히 올리고(워밍업) 이후 코사인으로 줄임
    steps_per_epoch = max(1, len(tr_dl))
    total, warm = a.epochs * steps_per_epoch, steps_per_epoch

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    start_epoch, best_f1, bad, history = 0, -1.0, 0, []
    last_path, best_path = os.path.join(out_dir, "last.pt"), os.path.join(out_dir, "best.pt")
    if a.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best_f1, bad, history = ck["epoch"] + 1, ck["best_f1"], ck["bad"], ck["history"]
        print(f"{start_epoch}에폭부터 이어서 학습")

    for epoch in range(start_epoch, a.epochs):
        head_only = a.train_mode == "head" or (a.model != "small_cnn" and epoch < a.freeze_epochs)
        set_trainable(model, head_only)
        t0 = time.time()
        tr_loss, tr_m = run_epoch(model, tr_dl, device, criterion, optimizer, scheduler, scaler)
        va_loss, va_m = run_epoch(model, va_dl, device, criterion)
        row = {"epoch": epoch + 1, "head_only": head_only, "train_loss": tr_loss, "train_acc": tr_m["acc"],
               "val_loss": va_loss, "val_acc": va_m["acc"], "val_off1_acc": va_m["off1_acc"],
               "val_macro_f1": va_m["macro_f1"], "lr": optimizer.param_groups[0]["lr"], "sec": time.time() - t0}
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
        print(f"[{epoch + 1:2d}/{a.epochs}] 학습 손실 {tr_loss:.3f} 정확도 {tr_m['acc']:.3f} | "
              f"검증 손실 {va_loss:.3f} 정확도 {va_m['acc']:.3f} 한칸 {va_m['off1_acc']:.3f} "
              f"F1 {va_m['macro_f1']:.3f} | {row['sec']:.0f}초")

        if va_m["macro_f1"] > best_f1:
            best_f1, bad = va_m["macro_f1"], 0
            torch.save({"model": model.state_dict(), "model_name": a.model, "epoch": epoch + 1}, best_path)
            print("  → 가장 좋은 모델 저장")
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                    "best_f1": best_f1, "bad": bad, "history": history}, last_path)
        if bad >= a.patience:
            print(f"검증 F1이 {a.patience}번 연속 오르지 않아 조기 종료")
            break

    print(f"끝. 최고 검증 macro-F1 {best_f1:.3f} → {best_path}")


if __name__ == "__main__":
    main()
