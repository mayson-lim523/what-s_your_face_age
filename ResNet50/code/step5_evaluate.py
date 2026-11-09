"""5단계: 학습된 모델 평가와 그래프 만들기.

- 비교 실험 단계에서는 --split-csv 에 검증 데이터(val_small.csv 또는 val.csv)를 넣는다.
- 시험 데이터(test.csv)는 최종 모델에서 딱 한 번만 넣는다.

사용 예:
python step5_evaluate.py --exp-dir results/base_resnet50 --split-csv splits/val_small.csv --image-root processed --tag val

결과 (results/<실험>/eval_<tag>/):
  metrics.json         정확도, 한 칸 차이 정확도, macro-F1
  per_band.csv         나이대별 정밀도·재현율·F1
  per_group.csv        출처별·인종 그룹별 정확도
  confusion_matrix.png 혼동 행렬
  learning_curve.png   학습 곡선 (history.csv로 그림)
  predictions.csv      사진마다 정답과 예측

여러 실험 비교표:
python step5_evaluate.py --summary results --tag val   → results/summary_val.csv
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import BANDS, NUM_CLASSES, compute_metrics


def set_korean_font():
    from matplotlib import font_manager
    for name in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]:
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_learning_curve(hist_csv, out_png):
    h = pd.read_csv(hist_csv)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["epoch"], h["train_loss"], label="학습")
    ax[0].plot(h["epoch"], h["val_loss"], label="검증")
    ax[0].set_title("손실 (낮을수록 좋음)"); ax[0].set_xlabel("에폭"); ax[0].legend()
    ax[1].plot(h["epoch"], h["train_acc"], label="학습 정확도")
    ax[1].plot(h["epoch"], h["val_acc"], label="검증 정확도")
    ax[1].plot(h["epoch"], h["val_macro_f1"], label="검증 macro-F1", linestyle="--")
    ax[1].set_title("점수 (높을수록 좋음)"); ax[1].set_xlabel("에폭"); ax[1].legend()
    from matplotlib.ticker import MaxNLocator
    for x in ax:
        x.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def plot_confusion(y, p, out_png):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, p, labels=list(range(NUM_CLASSES)))
    cm_norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(NUM_CLASSES), BANDS, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES), BANDS)
    ax.set_xlabel("예측한 나이대"); ax.set_ylabel("실제 나이대")
    ax.set_title("혼동 행렬 (행 기준 비율, 괄호는 장수)")
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def evaluate(a):
    import torch
    from torch.utils.data import DataLoader
    from common import FaceDataset, build_model, build_transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(a.exp_dir, "best.pt"), map_location=device, weights_only=False)
    model = build_model(ck["model_name"], pretrained=False).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    df = pd.read_csv(a.split_csv).reset_index(drop=True)
    dl = DataLoader(FaceDataset(df, a.image_root, build_transforms(train=False)), 128, num_workers=a.workers)
    probs = []
    with torch.no_grad():
        for x, _ in dl:
            probs.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
    probs = np.concatenate(probs)
    y, p = df["band"].astype(int).to_numpy(), probs.argmax(1)

    out = os.path.join(a.exp_dir, f"eval_{a.tag}")
    os.makedirs(out, exist_ok=True)
    m = compute_metrics(y, p)
    m.update({"split_csv": a.split_csv, "n": int(len(y)), "best_epoch": ck.get("epoch")})
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)

    from sklearn.metrics import precision_recall_fscore_support
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=list(range(NUM_CLASSES)), zero_division=0)
    pd.DataFrame({"나이대": BANDS, "정밀도": pr, "재현율": rc, "F1": f1, "장수": sup}).round(3) \
        .to_csv(os.path.join(out, "per_band.csv"), index=False, encoding="utf-8-sig")

    df["pred"] = p
    df["correct"] = (df["pred"] == df["band"]).astype(int)
    df["off1"] = ((df["pred"] - df["band"]).abs() <= 1).astype(int)
    groups = []
    for col in ["source", "race_group"]:
        if col in df:
            g = df.groupby(col).agg(장수=("correct", "size"), 정확도=("correct", "mean"), 한칸차이=("off1", "mean"))
            g.index = [f"{col}={i}" for i in g.index]
            groups.append(g)
    if groups:
        pd.concat(groups).round(3).to_csv(os.path.join(out, "per_group.csv"), encoding="utf-8-sig")
    df.to_csv(os.path.join(out, "predictions.csv"), index=False)

    set_korean_font()
    plot_confusion(y, p, os.path.join(out, "confusion_matrix.png"))
    hist = os.path.join(a.exp_dir, "history.csv")
    if os.path.exists(hist):
        plot_learning_curve(hist, os.path.join(out, "learning_curve.png"))
    print(json.dumps(m, ensure_ascii=False, indent=2))
    print("저장 폴더:", out)


def summary(a):
    rows = []
    for mpath in sorted(glob.glob(os.path.join(a.summary, "*", f"eval_{a.tag}", "metrics.json"))):
        exp = mpath.split(os.sep)[-3]
        m = json.load(open(mpath, encoding="utf-8"))
        cfg_path = os.path.join(a.summary, exp, "config.json")
        cfg = json.load(open(cfg_path, encoding="utf-8")) if os.path.exists(cfg_path) else {}
        rows.append({"실험": exp, "모델": cfg.get("model"), "학습방식": cfg.get("train_mode"), "학습률": cfg.get("lr"),
                     "증강": not cfg.get("no_augment", False), "제외데이터": cfg.get("exclude_source", ""),
                     "정확도": m["acc"], "한칸차이정확도": m["off1_acc"], "macroF1": m["macro_f1"]})
    t = pd.DataFrame(rows).round({"정확도": 3, "한칸차이정확도": 3, "macroF1": 3})
    path = os.path.join(a.summary, f"summary_{a.tag}.csv")
    t.to_csv(path, index=False, encoding="utf-8-sig")
    print(t.to_string(index=False))
    print("저장:", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir")
    ap.add_argument("--split-csv")
    ap.add_argument("--image-root")
    ap.add_argument("--tag", default="val")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--summary", help="결과 폴더 전체 비교표 만들기")
    args = ap.parse_args()
    summary(args) if args.summary else evaluate(args)
