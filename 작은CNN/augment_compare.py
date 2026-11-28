# 비교 실험 3 — 증강 ON vs OFF 정리 (표 + 그림)
#   입력: 작은 CNN history·평가 결과(로컬 results/), 팀원 v2 history(저장소 ConvNeXt/·ResNet50/results/)
#   출력: --out 폴더에 compare_table.csv/.md, curves_on_off.png, confusion_noaug_test.png, recall_on_off_test.png
#   실행(로컬 프로젝트 폴더에서):
#     python 작은CNN\augment_compare.py --repo C:\3th_project\what-s_your_face_age --out ...\작은CNN\results\no_aug\compare
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

BANDS = ["0-9세", "10대", "20대", "30대", "40대", "50대", "60세 이상"]
ON, OFF = "#257de0", "#f47230"                           # 증강 ON 파랑 / OFF 주황 (dataviz 기본 팔레트 1·2번)

p = argparse.ArgumentParser()
p.add_argument("--repo", required=True)                  # 팀 저장소 폴더 (팀원 history 위치)
p.add_argument("--on", default="results/small_cnn_v2_lr1e-3")   # 작은 CNN 증강 ON 결과 폴더
p.add_argument("--off", default="results/no_aug")        # 작은 CNN 증강 OFF 결과 폴더
p.add_argument("--out", required=True)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

def hist(path):
    return pd.read_csv(path)

def test_metrics(run):                                   # 작은 CNN만 test 평가가 있음
    with open(os.path.join(run, "eval_test", "metrics.json"), encoding="utf-8") as f:
        return json.load(f)

# ---------- 비교표 (val 기준: history에서 val macro-F1 최고 에폭 행) ----------
runs = [("작은 CNN", "OFF", os.path.join(a.off, "small_cnn_history.csv"), a.off),
        ("작은 CNN", "ON", os.path.join(a.on, "small_cnn_history.csv"), a.on),
        ("ResNet-50", "OFF", os.path.join(a.repo, "ResNet50/results/v2_noaug_resnet50_history.csv"), None),
        ("ResNet-50", "ON", os.path.join(a.repo, "ResNet50/results/v2_resnet50_history.csv"), None),
        ("ConvNeXt-Tiny", "OFF", os.path.join(a.repo, "ConvNeXt/results/v2_noaug_convnext_history.csv"), None),
        ("ConvNeXt-Tiny", "ON", os.path.join(a.repo, "ConvNeXt/results/v2_convnext_history.csv"), None)]
rows = []
for model, aug, hp, run in runs:
    h = hist(hp)
    b = h.loc[h["val_macro_f1"].idxmax()]               # 첫 최고값 (팀원 "클 때 저장"과 같음)
    last = h.iloc[-1]
    r = {"모델": model, "증강": aug,
         "val 정확도": round(b["val_acc"], 3), "val 한 칸 차이 정확도": round(b["val_off1_acc"], 3),
         "val macro-F1": round(b["val_macro_f1"], 3), "best 에폭": int(b["epoch"]),
         "마지막 에폭 val−train loss": round(last["val_loss"] - last["train_loss"], 3)}
    if run:
        t = test_metrics(run)
        r.update({"test 정확도": round(t["acc"], 3), "test 한 칸 차이 정확도": round(t["off1_acc"], 3),
                  "test macro-F1": round(t["macro_f1"], 3)})
    rows.append(r)
table = pd.DataFrame(rows)
table.to_csv(os.path.join(a.out, "compare_table.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(a.out, "compare_table.md"), "w", encoding="utf-8") as f:
    t = table.astype(object).fillna("—").astype(str)
    f.write("| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n")
    f.writelines("| " + " | ".join(r) + " |\n" for r in t.values)
print(table.astype(object).fillna("—").to_string(index=False))

# ---------- 그림 1: 작은 CNN 증강 ON/OFF 학습 곡선 ----------
hon, hoff = hist(os.path.join(a.on, "small_cnn_history.csv")), hist(os.path.join(a.off, "small_cnn_history.csv"))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for h, c, lab, dy in [(hon, ON, "증강 ON", -16), (hoff, OFF, "증강 OFF", 8)]:
    ax[0].plot(h["epoch"], h["train_loss"], color=c, lw=2, ls="--", label=f"{lab} train")
    ax[0].plot(h["epoch"], h["val_loss"], color=c, lw=2, label=f"{lab} val")
    ax[1].plot(h["epoch"], h["val_macro_f1"], color=c, lw=2, marker="o", ms=4, label=lab)
    bi = h["val_macro_f1"].idxmax()
    ax[1].annotate(f"best {h['val_macro_f1'][bi]:.3f} (ep {int(h['epoch'][bi])})",
                   (h["epoch"][bi], h["val_macro_f1"][bi]), textcoords="offset points", xytext=(-40, dy), fontsize=9, color="#444")
ax[0].set(title="손실 (점선 train, 실선 val)", xlabel="에폭", ylabel="loss")
ax[1].set(title="val macro-F1", xlabel="에폭", ylabel="macro-F1")
for x in ax:
    x.grid(alpha=0.3); x.legend(frameon=False); x.spines[["top", "right"]].set_visible(False)
fig.suptitle("작은 CNN — 증강 ON vs OFF (v2 데이터, 20에폭)")
fig.tight_layout(); fig.savefig(os.path.join(a.out, "curves_on_off.png"), dpi=150); plt.close(fig)

# ---------- 그림 2: 증강 OFF 혼동 행렬 (test, 행 정규화) ----------
pr = pd.read_csv(os.path.join(a.off, "eval_test", "predictions.csv"))
cm = confusion_matrix(pr["band"], pr["pred"], labels=range(7)).astype(float)
cm /= cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
for i in range(7):
    for j in range(7):
        ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=9,
                color="white" if cm[i, j] > 0.5 else "black")
ax.set_xticks(range(7), BANDS, rotation=30); ax.set_yticks(range(7), BANDS)
ax.set(xlabel="예측", ylabel="정답", title="작은 CNN 증강 OFF — 혼동 행렬 (test, 행 정규화)")
fig.colorbar(im, fraction=0.046); fig.tight_layout()
fig.savefig(os.path.join(a.out, "confusion_noaug_test.png"), dpi=150); plt.close(fig)

# ---------- 그림 3: 나이대별 recall (test) ON vs OFF ----------
ron = pd.read_csv(os.path.join(a.on, "eval_test", "per_band.csv"))["재현율"].values
roff = pd.read_csv(os.path.join(a.off, "eval_test", "per_band.csv"))["재현율"].values
x = np.arange(7); w = 0.38
fig, ax = plt.subplots(figsize=(9, 4.5))
for off, vals, c, lab in [(-w / 2, ron, ON, "증강 ON"), (w / 2, roff, OFF, "증강 OFF")]:
    bars = ax.bar(x + off, vals, w - 0.02, color=c, label=lab)
    ax.bar_label(bars, fmt="%.2f", fontsize=8)
ax.set_xticks(x, BANDS); ax.set_ylim(0, 1); ax.set_ylabel("recall")
ax.set_title("작은 CNN 나이대별 recall (test) — 증강 ON vs OFF")
ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(a.out, "recall_on_off_test.png"), dpi=150); plt.close(fig)
pd.DataFrame({"나이대": BANDS, "recall ON": ron, "recall OFF": roff, "차이(OFF−ON)": roff - ron}).round(3) \
    .to_csv(os.path.join(a.out, "recall_on_off_test.csv"), index=False, encoding="utf-8-sig")
print("저장:", a.out)
