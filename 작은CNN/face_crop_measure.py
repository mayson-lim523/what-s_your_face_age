# #############################################################################
#
#   YuNet 상자 기준 여백 측정 — 학습 사진(training)이 YuNet 상자에서 얼마나 떨어져 잘렸는지 거꾸로 재기
#   인공지능 4조 · 담당: LIM
#
#   [방법]
#     - splits_csv_v2/train.csv에서 FairFace · UTKFace 각 1,000장 (시드 42). AAF는 넓은 크롭이라 제외
#     - face_align.locate로 얼굴 상자를 찾음 (테두리 덧대기 포함, 정렬 끔/켬 두 번)
#     - 여백 = (원래 사진 테두리 − 상자) ÷ 상자 크기, 왼·위·오른·아래. 정렬 켬이면 원래 사진이 회전되므로
#       "회전된 원래 사진의 중심에서 가로·세로 224" 정사각형을 기준으로 잰다
#     - 최종 여백 = 출처별 중앙값을 학습 데이터 비율 FairFace 70 : UTKFace 19로 가중 평균
#     - 재크롭 IoU = 최종 여백으로 다시 자른 상자 vs 원래 사진이 캔버스에서 차지하는 영역(정렬 켬이면 회전된 사각형)
#
#   [실행] 저장소 최상위에서:
#     .venv\Scripts\python 작은CNN\face_crop_measure.py --image-root C:\3th_project\whats_your_face_age\division_224AFAD-X
#   [결과] 작은CNN/results/face_crop/  margins_per_image.csv, margins_summary.csv, crop_examples.png
#
# #############################################################################

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import face_align as fa  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WEIGHT = {"FairFace": 70, "UTKFace": 19}
N_PER_SOURCE, SEED = 1000, 42
plt.rcParams["font.family"] = "Malgun Gothic"


def source_of(name):
    if name.endswith(".jpg.chip.jpg"):
        return "UTKFace"
    if name[:-4].isdigit():
        return "FairFace"
    return "AAF/기타"


# =============================================================================
#   ▶ [모듈 1] 사진 1장 측정 — 상자, 여백 4개, 원래 사진 영역(캔버스 좌표 꼭짓점 4개)
# =============================================================================

def measure_one(rgb, align):
    loc = fa.locate(rgb, align)
    if loc is None:
        return None
    H, W = rgb.shape[:2]
    p, M = loc["pad"], loc["M"]
    corners = np.array([[p, p], [p + W, p], [p + W, p + H], [p, p + H]], np.float64)
    corners = (M[:, :2] @ corners.T).T + M[:, 2]
    cx, cy = corners.mean(0)
    x, y, w, h = loc["box"]
    m = ((x - (cx - W / 2)) / w, (y - (cy - H / 2)) / h, ((cx + W / 2) - (x + w)) / w, ((cy + H / 2) - (y + h)) / h)
    return dict(box=loc["box"], margins=m, corners=corners, shape=loc["canvas"].shape[:2], canvas=loc["canvas"],
                n_faces=loc["n_faces"])


def iou(r, corners, shape):
    """잘라낼 상자(캔버스 안으로 자름) vs 원래 사진 영역 다각형 — 캔버스 크기 마스크로 계산"""
    a = np.zeros(shape, np.uint8)
    b = np.zeros(shape, np.uint8)
    l, t, rr, bb = (int(round(v)) for v in r)
    a[max(t, 0):max(bb, 0), max(l, 0):max(rr, 0)] = 1
    cv2.fillPoly(b, [np.round(corners).astype(np.int32)], 1)
    return (a & b).sum() / max((a | b).sum(), 1)


# =============================================================================
#   ▶ [모듈 2] 전체 측정 → 표
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results" / "face_crop")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ROOT / "splits_csv_v2" / "train.csv")
    df["source"] = df.path.map(lambda s: source_of(Path(s).name))
    sample = pd.concat([df[df.source == s].sample(N_PER_SOURCE, random_state=SEED) for s in WEIGHT])

    rows, keep = [], {}
    for path, src in zip(sample.path, sample.source):
        rgb = np.asarray(Image.open(a.image_root / path).convert("RGB"))
        for align in (False, True):
            r = measure_one(rgb, align)
            rows.append(dict(path=path, source=src, align=align, found=r is not None,
                             **({} if r is None else dict(zip(["ml", "mt", "mr", "mb"], r["margins"])))))
            if r is not None:
                keep[(path, align)] = r
    per = pd.DataFrame(rows)

    # 출처·정렬별 중앙값 → 70:19 가중 → 최종 여백
    med = per[per.found].groupby(["source", "align"])[["ml", "mt", "mr", "mb"]].median()
    final = {al: tuple(sum(WEIGHT[s] * med.loc[(s, al)].to_numpy() for s in WEIGHT) / sum(WEIGHT.values()))
             for al in (False, True)}

    # IoU: 출처별 자기 여백 / 최종 여백 두 가지
    def iou_of(row, m):
        r = keep[(row.path, row.align)]
        return iou(fa.crop_rect(r["box"], m), r["corners"], r["shape"])
    ok = per[per.found].copy()
    ok["iou_own"] = [iou_of(r, tuple(med.loc[(r.source, r.align)])) for r in ok.itertuples()]
    ok["iou_final"] = [iou_of(r, final[r.align]) for r in ok.itertuples()]
    per = per.merge(ok[["path", "align", "iou_own", "iou_final"]], on=["path", "align"], how="left")
    per.to_csv(a.out_dir / "margins_per_image.csv", index=False, encoding="utf-8-sig")

    summ = per.groupby(["source", "align"]).agg(
        n=("found", "size"), det_rate=("found", "mean"), ml=("ml", "median"), mt=("mt", "median"),
        mr=("mr", "median"), mb=("mb", "median"), iou_own=("iou_own", "median"), iou_final=("iou_final", "median"))
    summ.to_csv(a.out_dir / "margins_summary.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(summ.round(3))
    for al in (False, True):
        w_iou = sum(WEIGHT[s] * summ.loc[(s, al), "iou_final"] for s in WEIGHT) / sum(WEIGHT.values())
        w_det = sum(WEIGHT[s] * summ.loc[(s, al), "det_rate"] for s in WEIGHT) / sum(WEIGHT.values())
        print(f"정렬 {'켬' if al else '끔'}: 최종 여백 {tuple(round(v, 3) for v in final[al])}, "
              f"가중 IoU 중앙값 {w_iou:.3f}, 가중 검출률 {w_det:.3f}")

    # =========================================================================
    #   ▶ [모듈 3] 크롭 예시 6쌍 (출처별 3장, 정렬 끔/켬 둘 다) — [원본 | 다시 자름(끔) | 다시 자름(켬)]
    # =========================================================================
    rng = np.random.default_rng(SEED)
    picks = []
    for s in WEIGHT:
        cand = ok[(ok.source == s) & ~ok["align"]].path.unique()
        picks += [(s, p) for p in rng.choice(cand, 3, replace=False)]
    fig, axes = plt.subplots(6, 3, figsize=(7.5, 15))
    for i, (s, p) in enumerate(picks):
        img = Image.open(a.image_root / p).convert("RGB")
        ims = [img] + [fa.crop_face(img, align=al, margins=final[al])[0] for al in (False, True)]
        for j, (im, t) in enumerate(zip(ims, ["원본 학습 사진", "YuNet 다시 자름 (정렬 끔)", "YuNet 다시 자름 (정렬 켬)"])):
            ax = axes[i, j]
            ax.imshow(im if im is not None else np.zeros((224, 224, 3), np.uint8))
            ax.set_xticks([]), ax.set_yticks([])
            if i == 0:
                ax.set_title(t, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{s}\n{Path(p).name[:18]}", fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out_dir / "crop_examples.png", dpi=110)
    print("저장:", a.out_dir)


if __name__ == "__main__":
    main()
