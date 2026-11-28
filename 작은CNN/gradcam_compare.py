# #############################################################################
#
#   세 모델 Grad-CAM 비교 — 작은 CNN · ConvNeXt-Tiny · ResNet-50 (검증 데이터만)
#   인공지능 4조 · 담당: LIM
#
#   [하는 일]
#     P1 세 모델로 val 전체 예측 (샘플 고르기용, predictions_3models.csv)
#     P2 나이대마다 "세 모델 모두 맞힌" 3장 + "세 모델 모두 틀린" 3장 → [원본 | 작은CNN | ConvNeXt | ResNet] 격자
#     P3 나이대마다 200장 → 모델별 히트맵 에너지의 가운데 60% 상자 / 가장자리 비율 (나이대별, 출처별) + 평균 히트맵
#     P4 출처(파일 이름 규칙)별 사진 테두리 특징: 가장자리 8픽셀 띠 밝기·표준편차, 검은 테두리 비율
#
#   [실행] 저장소 최상위에서 (CPU 권장: GPU 작업 중 PC가 멈춘 적이 있음)
#     .venv\Scripts\python 작은CNN\gradcam_compare.py --image-root C:\3th_project\whats_your_face_age\division_224AFAD-X
#
#   시험 데이터(test.csv)는 쓰지 않는다 — --csv 이름에 test가 들어 있으면 멈춤
#
# #############################################################################

import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from gradcam_utils import BANDS, MODEL_NAMES, GradCAM, center_mask, center_ratio, load_model, overlay, preprocess

SHORT = {"small_cnn": "작은CNN", "convnext_tiny": "ConvNeXt", "resnet50": "ResNet50"}
SOURCES = ["FairFace", "UTKFace", "AAF", "기타"]
SEED = 42


# =============================================================================
#   ▶ [P0] 준비 — 옵션, 검증 CSV, 출처 붙이기
#   - CSV의 source 열은 전부 "unknown"이라 쓸 수 없음 → 팀 전처리 코드의 파일 이름 규칙으로 출처를 정함
#     (개발일지 17, source_check.py와 같은 규칙: 숫자A나이 = AAF, .jpg.chip.jpg = UTKFace, 숫자.jpg = FairFace)
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="세 모델 Grad-CAM 비교 (val만)")
    p.add_argument("--csv", default=os.path.join(os.path.dirname(__file__), "..", "splits_csv_v2", "val.csv"))
    p.add_argument("--image-root", required=True)
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "results", "gradcam_compare"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--n-energy", type=int, default=200)   # P3 나이대별 장수
    p.add_argument("--n-sample", type=int, default=3)     # P2 나이대별 맞힘/틀림 각 장수
    a = p.parse_args()
    if "test" in os.path.basename(a.csv).lower():
        raise SystemExit("test.csv는 팀이 최종 모델을 정한 뒤 한 번만 씁니다. 멈춥니다.")
    return a


def source_of(path):
    name = os.path.basename(path)
    if re.fullmatch(r"\d+A\d+\.jpg", name):
        return "AAF"
    if name.endswith(".jpg.chip.jpg"):
        return "UTKFace"
    if re.fullmatch(r"\d+\.jpg", name):
        return "FairFace"
    return "기타"


def load_images(root, paths):
    return [Image.open(os.path.join(root, p)).convert("RGB") for p in paths]


# =============================================================================
#   ▶ [P1] 세 모델 val 전체 예측
#   - 샘플을 "세 모델 모두 맞힘/틀림"으로 고르려면 전체 예측이 먼저 필요
#   - 한 번 만들면 CSV로 저장해 두고 다시 실행할 때 재사용 (CPU로 수 분 걸림)
#   - 정확도·macro-F1을 학습 기록과 비교해서 가중치·전처리가 맞게 읽혔는지 확인
# =============================================================================

def predict_all(df, models, a):
    path = os.path.join(a.out_dir, "predictions_3models.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    out = df.copy()
    for name, model in models.items():
        preds = []
        with torch.no_grad():
            for s in range(0, len(df), 64):
                x = preprocess(load_images(a.image_root, df["path"].iloc[s:s + 64])).to(a.device)
                preds.append(model(x).argmax(1).cpu().numpy())
        out[f"pred_{name}"] = np.concatenate(preds)
        print(f"  {name} 예측 완료")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


# =============================================================================
#   ▶ [P2] 샘플 격자 — 같은 사진에 세 모델 히트맵을 나란히
#   - 맞힘 = 세 모델 모두 정답, 틀림 = 세 모델 모두 오답. 그래야 "같은 사진"에서 모델 차이만 비교됨
#   - 히트맵은 각 모델이 예측한 나이대 기준 (모델이 그 답을 고를 때 어디를 봤는지)
# =============================================================================

def part2_samples(pred, cams, a):
    rng = np.random.default_rng(SEED)
    ok = np.all([pred[f"pred_{m}"] == pred["band"] for m in MODEL_NAMES], axis=0)
    bad = np.all([pred[f"pred_{m}"] != pred["band"] for m in MODEL_NAMES], axis=0)
    rows = []
    for b in range(len(BANDS)):
        for kind, mask in (("맞힘", ok), ("틀림", bad)):
            idx = np.flatnonzero(mask & (pred["band"].values == b))
            for i in rng.choice(idx, min(a.n_sample, len(idx)), replace=False):
                rows.append({"band": b, "kind": kind, "i": int(i)})
    sel = pd.DataFrame(rows)
    sel = sel.join(pred.iloc[sel["i"]].reset_index(drop=True).drop(columns="band"))
    sel.to_csv(os.path.join(a.out_dir, "samples.csv"), index=False, encoding="utf-8-sig")

    for b in range(len(BANDS)):
        s = sel[sel["band"] == b].reset_index(drop=True)
        imgs = load_images(a.image_root, s["path"])
        x = preprocess(imgs)
        res = {m: cams[m](x) for m in MODEL_NAMES}
        fig, axes = plt.subplots(len(s), 4, figsize=(9, 2.4 * len(s)))
        for r in range(len(s)):
            axes[r, 0].imshow(imgs[r].resize((224, 224)))
            axes[r, 0].set_title(f"[{s.kind[r]}] 정답 {BANDS[b]}\n{s.source[r]}", fontsize=8)
            for c, m in enumerate(MODEL_NAMES, 1):
                heat, cls, prob = res[m]
                axes[r, c].imshow(overlay(imgs[r], heat[r]))
                mark = "맞음" if cls[r] == b else "틀림"
                axes[r, c].set_title(f"{SHORT[m]}: {BANDS[cls[r]]} ({prob[r, cls[r]]:.2f}) {mark}", fontsize=8,
                                     color="black" if cls[r] == b else "crimson")
        for ax in axes.flat:
            ax.axis("off")
        fig.suptitle(f"{BANDS[b]} — 위 {sum(s.kind == '맞힘')}줄: 세 모델 모두 맞힘 / 아래 {sum(s.kind == '틀림')}줄: 세 모델 모두 틀림",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out_dir, f"samples_band{b}.png"), dpi=110)
        plt.close(fig)
    print("  샘플 격자 7장 저장")
    return sel


# =============================================================================
#   ▶ [P3] 히트맵 에너지 — 가운데 60% 상자 vs 가장자리
#   - 나이대마다 200장을 시드 42로 무작위 추출 (맞힘·틀림 섞임 = 실제 사용 상황과 같게)
#   - 사진별 비율을 낸 뒤 평균. 고르게 퍼진 히트맵이면 가운데 35.8%(= 상자 면적)가 기준선
#   - 평균 히트맵(사진별 0~1 히트맵의 평균)도 그려서 "어디에 몰리는지" 모양으로 확인
# =============================================================================

def part3_energy(pred, cams, a):
    rng = np.random.default_rng(SEED)
    idx = np.concatenate([rng.choice(np.flatnonzero(pred["band"].values == b), a.n_energy, replace=False)
                          for b in range(len(BANDS))])
    sub = pred.iloc[idx].reset_index(drop=True)
    mean_heat = {m: np.zeros((224, 224)) for m in MODEL_NAMES}
    for m in MODEL_NAMES:
        ratios = []
        for s in range(0, len(sub), a.batch):
            heat, _, _ = cams[m](preprocess(load_images(a.image_root, sub["path"].iloc[s:s + a.batch])))
            ratios.append(center_ratio(heat))
            mean_heat[m] += heat.sum(0)
        sub[f"center_{m}"] = np.concatenate(ratios).astype(float)
        mean_heat[m] /= len(sub)
        print(f"  {m} 에너지 계산 완료 (히트맵 전부 0인 사진 {sub[f'center_{m}'].isna().sum()}장 제외)")
    sub.to_csv(os.path.join(a.out_dir, "energy_per_image.csv"), index=False, encoding="utf-8-sig")

    cols = [f"center_{m}" for m in MODEL_NAMES]
    rename = {f"center_{m}": SHORT[m] for m in MODEL_NAMES}
    by_band = (sub.groupby("band")[cols].mean() * 100).rename(index=dict(enumerate(BANDS)), columns=rename)
    by_band.loc["전체"] = sub[cols].mean().values * 100
    by_src = (sub.groupby("source")[cols].mean() * 100).rename(columns=rename).reindex([s for s in SOURCES if s in set(sub.source)])
    by_src["장수"] = sub.groupby("source").size()
    by_src_band = sub.pivot_table(index="band", columns="source", values="center_small_cnn", aggfunc="mean") * 100
    by_src_band.index = BANDS
    for t, name in ((by_band, "energy_by_band.csv"), (by_src, "energy_by_source.csv"), (by_src_band, "energy_smallcnn_band_x_source.csv")):
        t.round(1).to_csv(os.path.join(a.out_dir, name), encoding="utf-8-sig")
    print("\n[가운데 60% 상자 에너지 %] 나이대별\n", by_band.round(1).to_string())
    print("\n[가운데 60% 상자 에너지 %] 출처별\n", by_src.round(1).to_string())
    print("\n[작은CNN 가운데 에너지 %] 나이대 × 출처\n", by_src_band.round(1).to_string())

    # 평균 히트맵 그림 (세 모델, 같은 1,400장)
    lo = round(224 * 0.2)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for ax, m in zip(axes, MODEL_NAMES):
        ax.imshow(mean_heat[m], cmap="jet", vmin=0, vmax=mean_heat[m].max())
        ax.add_patch(patches.Rectangle((lo, lo), 224 - 2 * lo, 224 - 2 * lo, fill=False, ec="white", lw=1.5, ls="--"))
        ax.set_title(f"{SHORT[m]} — 가운데 {by_band.loc['전체', SHORT[m]]:.1f}%", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"평균 Grad-CAM (val, 나이대별 {a.n_energy}장 × 7 = {len(sub):,}장) · 점선 = 가운데 60% 상자 (면적 35.8%)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "mean_heatmap_3models.png"), dpi=130)
    plt.close(fig)
    return sub


# =============================================================================
#   ▶ [P4] 출처별 사진 테두리 특징 (val 전체)
#   - 개발일지 16과 같은 정의: 가장자리 8픽셀 띠의 밝기 평균·표준편차(흑백 0~255),
#     검은 테두리 = 띠 픽셀의 절반 이상이 밝기 15 미만
#   - 출처마다 사진을 자른 방식이 다르면 테두리 특징이 다르게 나옴 → 모델이 출처(=나이대 비율 차이)를 알아챌 단서
# =============================================================================

def part4_border(pred, a):
    ring = ~center_mask(1 - 2 * 8 / 224)            # 가장자리 8픽셀 띠
    rows = []
    for p in pred["path"]:
        g = np.asarray(Image.open(os.path.join(a.image_root, p)).convert("L").resize((224, 224)), dtype=np.float32)[ring]
        rows.append((float(g.mean()), float(g.std()), (g < 15).mean() >= 0.5))
    b = pd.DataFrame(rows, columns=["edge_mean", "edge_std", "black_border"])
    b["source"] = pred["source"].values
    t = b.groupby("source").agg(장수=("edge_mean", "size"), 가장자리밝기=("edge_mean", "mean"),
                                가장자리표준편차=("edge_std", "mean"), 검은테두리비율=("black_border", "mean"))
    t["검은테두리비율"] *= 100
    t = t.reindex([s for s in SOURCES if s in t.index]).round(1)
    t.to_csv(os.path.join(a.out_dir, "border_by_source.csv"), encoding="utf-8-sig")
    print("\n[출처별 테두리 특징] val 전체\n", t.to_string())
    return t


if __name__ == "__main__":
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))   # CPU 코어 2개는 남겨서 PC가 멈추지 않게

    df = pd.read_csv(a.csv)
    df["source"] = df["path"].map(source_of)
    print(f"검증 {len(df):,}장 | 출처 {df.source.value_counts().to_dict()} | 장치 {a.device}")
    models = {m: load_model(m, device=a.device) for m in MODEL_NAMES}

    pred = predict_all(df, models, a)
    for m in MODEL_NAMES:
        acc = (pred[f"pred_{m}"] == pred["band"]).mean()
        f1 = f1_score(pred["band"], pred[f"pred_{m}"], average="macro", labels=list(range(len(BANDS))), zero_division=0)
        print(f"  {m}: val 정확도 {acc:.4f}, macro-F1 {f1:.4f}")

    cams = {m: GradCAM(models[m], m) for m in MODEL_NAMES}
    part2_samples(pred, cams, a)
    part3_energy(pred, cams, a)
    part4_border(pred, a)
    print(f"\n끝 → {a.out_dir}")
