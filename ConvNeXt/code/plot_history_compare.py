"""두 실험의 history.csv를 나란히 그려 비교한다 (예: 증강 있음 vs 없음).

그래프 3개: ① 학습·검증 정확도  ② 검증 손실  ③ 검증 macro-F1 (최고 에폭 표시)

사용 예:
python plot_history_compare.py --a results/full20_resnet50 --b results/noaug_resnet50 --label-a "증강 있음" --label-b "증강 없음" --title "ResNet-50: 증강 있음 vs 없음" --out aug_compare_resnet50.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd



def set_korean_font():
    from matplotlib import font_manager
    for name in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]:
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False

C_A, C_B = "#2563eb", "#dc2626"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="첫 번째 실험 폴더 (history.csv가 있는 곳)")
    ap.add_argument("--b", required=True, help="두 번째 실험 폴더")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", default="history_compare.png")
    a = ap.parse_args()

    set_korean_font()
    runs = [(pd.read_csv(os.path.join(a.a, "history.csv")), C_A, a.label_a),
            (pd.read_csv(os.path.join(a.b, "history.csv")), C_B, a.label_b)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for h, c, lab in runs:
        axes[0].plot(h["epoch"], h["train_acc"], "--", color=c, lw=1.6, label=f"{lab} · 학습")
        axes[0].plot(h["epoch"], h["val_acc"], "-", color=c, lw=2.2, label=f"{lab} · 검증")
        axes[1].plot(h["epoch"], h["val_loss"], "-o", color=c, ms=3, lw=2, label=lab)
        axes[2].plot(h["epoch"], h["val_macro_f1"], "-o", color=c, ms=3, lw=2, label=lab)
        b = h["val_macro_f1"].idxmax()
        axes[2].annotate(f"최고 {h.loc[b, 'val_macro_f1']:.3f} ({int(h.loc[b, 'epoch'])}에폭)",
                         (h.loc[b, "epoch"], h.loc[b, "val_macro_f1"]), textcoords="offset points",
                         xytext=(25, 12 if c == C_A else -48), color=c, fontsize=9, fontweight="bold",
                         arrowprops=dict(arrowstyle="-", color=c, lw=0.8))
    lo = min(r[0]["val_macro_f1"].min() for r in runs)
    hi = max(r[0]["val_macro_f1"].max() for r in runs)
    axes[2].set_ylim(lo - 0.02, hi + 0.035)
    axes[0].set_title("정확도 (점선 = 학습, 실선 = 검증)")
    axes[1].set_title("검증 손실 (낮을수록 좋음)")
    axes[2].set_title("검증 macro-F1 (높을수록 좋음)")
    for ax in axes:
        ax.set_xlabel("에폭"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    if a.title:
        fig.suptitle(a.title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
