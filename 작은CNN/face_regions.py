# #############################################################################
#
#   Grad-CAM 히트맵 글 설명 — 얼굴 영역 나누기 · 영역별 주목 비중 · 가림 실험
#   인공지능 4조 · 담당: LIM
#
#   [쓰는 곳] app_gradio.py (Grad-CAM 아래 글 설명)
#   [중요] YuNet 5점은 "설명용"으로만 쓴다. 모델 입력(224 얼굴 사진)에는 아무것도 더하지 않는다
#   [점검·비교표] 저장소 최상위에서:
#     .venv\Scripts\python 작은CNN\face_regions.py --image-root <division_224AFAD-X 폴더>
#     → 작은CNN/results/face_regions/ 에 영역 경계 그림 + 비교표(compare.csv) 저장
#
# #############################################################################

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# 화면 기준 왼쪽/오른쪽 (사진을 보는 사람 기준. YuNet 5점의 "오른눈"은 화면 왼쪽에 있음)
REGIONS = ["이마", "왼쪽 눈가", "오른쪽 눈가", "코", "입", "왼쪽 볼", "오른쪽 볼", "턱", "기타(머리·배경)"]
OTHER = len(REGIONS) - 1
BLUR_RADIUS = 16          # 가림 실험 블러 세기 (224 사진 기준 가우시안 반지름, 눈·입 모양이 안 보일 정도)


# =============================================================================
#   ▶ [모듈 1] 영역 정의 — 모든 기준이 이 함수 하나에 있다
#   좌표계: 두 눈 가운데 E를 원점, 눈→눈 방향 u(가로), 그와 수직인 v(세로, 아래가 +)
#     a = 가로 위치 ÷ 눈 사이 거리 d,  b = 세로 위치 ÷ 눈-입 높이 h (눈 = 0, 입꼬리 가운데 = 1)
#     → 얼굴이 기울거나 크기가 달라도 같은 기준. 각 픽셀은 위에서부터 처음 맞는 영역 하나에만 들어감
#   ① 얼굴 타원 밖 → 기타(머리카락·배경). 타원 중심 (0, 0.35), 반지름 가로 1.05d · 세로 1.35h
#      (위 끝 b=-1.0 ≈ 이마 위, 아래 끝 b=1.7 ≈ 턱끝, 좌우 ≈ 볼 바깥)
#   ② 눈가: 각 눈 중심 타원, 반지름 가로 0.42d · 세로 0.32h (눈썹 아래 ~ 눈 밑 다크서클)
#   ③ 입: 입꼬리 가운데 중심 타원, 반지름 가로 (입 폭/2 + 0.2d) · 세로 0.3h
#   ④ 코: 코끝 세로줄 ±0.28d 안, 눈 높이(b=-0.1)부터 코끝 + 0.2h까지 (콧등 + 코끝 + 콧볼)
#   ⑤ 이마: b < -0.3 (눈썹 위)
#   ⑥ 턱: b > 1.3 (입 타원 아래)
#   ⑦ 볼: 나머지 얼굴, 코끝 세로줄 기준 왼쪽/오른쪽
# =============================================================================

def face_regions(lm, size=224):
    """lm: 5점 (5,2) — 오른눈, 왼눈, 코, 오른입꼬리, 왼입꼬리 (224 좌표) → 영역 번호 (size,size) uint8"""
    lm = np.asarray(lm, np.float64)
    E, M = (lm[0] + lm[1]) / 2, (lm[3] + lm[4]) / 2
    u = (lm[1] - lm[0]) / np.linalg.norm(lm[1] - lm[0])
    v = np.array([-u[1], u[0]])
    d = np.linalg.norm(lm[1] - lm[0])
    h = max((M - E) @ v, 1e-6)
    to_ab = lambda p: np.stack([(p - E) @ u / d, (p - E) @ v / h], -1)
    yy, xx = np.mgrid[:size, :size] + 0.5                       # 픽셀 중심
    a, b = np.moveaxis(to_ab(np.stack([xx, yy], -1)), -1, 0)
    (ar, _), (al, _), (an, bn), (amr, _), (aml, _) = to_ab(lm)
    am = (amr + aml) / 2

    ell = lambda ca, cb, ra, rb: ((a - ca) / ra) ** 2 + ((b - cb) / rb) ** 2 <= 1
    rules = [                                                    # (영역 번호, 조건) — 앞이 우선
        (OTHER, ~ell(0, 0.35, 1.05, 1.35)),
        (1, ell(ar, 0, 0.42, 0.32)),
        (2, ell(al, 0, 0.42, 0.32)),
        (4, ell(am, 1, abs(aml - amr) / 2 + 0.2, 0.3)),
        (3, (abs(a - an) <= 0.28) & (b >= -0.1) & (b <= bn + 0.2)),
        (0, b < -0.3),
        (7, b > 1.3),
        (5, a < an),
        (6, np.ones_like(a, bool)),
    ]
    if ar > al:                     # 오른눈이 화면 오른쪽에 있으면(뒤집힌 사진) 화면 기준 이름을 맞바꿈
        rules = [({1: 2, 2: 1, 5: 6, 6: 5}.get(k, k), c) for k, c in rules]
    out = np.full((size, size), 255, np.uint8)
    for k, cond in rules:
        out[(out == 255) & cond] = k
    return out


# =============================================================================
#   ▶ [모듈 2] 주목 비중 — 224 히트맵의 영역별 합 ÷ 전체 합 (%). 영역이 사진을 빈틈없이 나누므로 합 = 100
# =============================================================================

def attention_share(heat, labels):
    heat = np.asarray(heat, np.float64)
    total = heat.sum()
    if total <= 0:
        return np.full(len(REGIONS), np.nan)
    return 100 * np.bincount(labels.ravel(), heat.ravel(), len(REGIONS)) / total


def share_text(share, top=4):
    if np.isnan(share).any():
        return "주목 비중 — 히트맵이 비어 있음"
    order = np.argsort(-share)
    parts = [f"{REGIONS[i]} {share[i]:.0f}%" for i in order[:top]]
    return "주목 비중 — " + " · ".join(parts) + f" · 나머지 {share[order[top:]].sum():.0f}%"


# =============================================================================
#   ▶ [모듈 3] 가림 실험 — 영역 하나만 강하게 블러한 사진 9장을 한 배치로 예측
#   - 블러는 사진 전체를 블러한 뒤 그 영역 픽셀만 바꿔 넣음 (영역 밖은 원본 그대로)
#   - 반환: 영역별 가린 뒤 확률 (9,) — cls(예측 1위 나이대) 기준
# =============================================================================

def occluded_images(face, labels):
    base = np.asarray(face.convert("RGB"))
    blur = np.asarray(face.convert("RGB").filter(ImageFilter.GaussianBlur(BLUR_RADIUS)))
    return [Image.fromarray(np.where((labels == k)[..., None], blur, base)) for k in range(len(REGIONS))]


def occlusion(model, face, labels, cls, preprocess):
    import torch
    x = preprocess(occluded_images(face, labels)).to(next(model.parameters()).device)
    with torch.no_grad():
        return model(x).softmax(1)[:, cls].cpu().numpy()


def occlusion_text(p0, p, band):
    order = np.argsort(p - p0)                                 # 하락 큰 순
    rows = [f"| {REGIONS[k]} | {100 * p0:.0f}% → {100 * p[k]:.0f}% | {round(100 * (p[k] - p0)):+d}%p |" for k in order]
    return (f"**가림 실험** (예측 1위 {band} 확률, 하락 큰 순)\n\n| 가린 영역 | 확률 | 차이 |\n|---|---|---|\n"
            + "\n".join(rows) + "\n\n가린 사진은 학습 때 없던 형태라 수치가 과장될 수 있음")


# =============================================================================
#   ▶ [모듈 4] 디버그 그림 — 영역 경계선 + 5점 + 영역 이름
# =============================================================================

def draw_regions(face, labels, lm):
    img = face.convert("RGB").resize((448, 448), Image.BILINEAR)
    big = np.kron(labels, np.ones((2, 2), np.uint8))
    edge = (big[1:, 1:] != big[:-1, 1:]) | (big[1:, 1:] != big[1:, :-1])
    arr = np.asarray(img).copy()
    arr[1:, 1:][edge] = (255, 255, 0)
    img = Image.fromarray(arr)
    dr = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("malgun.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for k in range(len(REGIONS)):
        ys, xs = np.nonzero(big == k)
        if len(xs):
            xy = (4, 4) if k == OTHER else (np.median(xs) - 15, np.median(ys) - 8)   # 기타는 테두리 고리라 모서리에
            dr.text(xy, REGIONS[k].split("(")[0], fill=(255, 255, 255),
                    font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    for x, y in np.asarray(lm) * 2:
        dr.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 0, 0))
    return img


# =============================================================================
#   점검 + 비교표 — 시연 사진(val 사진을 1280×960 회색 잡음 배경에 300×300으로 붙인 합성 사진)으로
#   (배경을 같은 사진을 흐려 만들면 YuNet이 배경의 큰 흐린 얼굴을 잡는 경우가 있어 얼굴 없는 배경을 씀)
#   ① 정렬 끔/켬 좌표 확인: 반환한 5점을 원본 사진 좌표로 되돌린 자리에 색 점을 찍고 다시 자른 뒤,
#      224 사진에서 찾은 색 점 위치 vs 반환된 5점 (px). 색 점이 YuNet 검출을 흔들지 않게 색 점 사진에서는
#      원래 사진의 검출 결과를 그대로 다시 씀 → 순수하게 좌표 변환(회전·자르기·224로 늘림)만 검사
#      (224 사진을 다시 검출해 비교하면 YuNet이 해상도마다 5~10px 흔들려서 변환 오류와 구분이 안 됨)
#   ② 영역 경계 그림 저장 ③ 주목 비중 합 = 100% ④ 세 모델 주목 비중 1위 vs 가림 실험 하락 1위
#   정렬 켬 확인에는 15도 기울인 사진도 넣음. test.csv는 쓰지 않음
# =============================================================================
if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    import pandas as pd

    HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(HERE))
    import face_align as fa
    from gradcam_utils import BANDS, GradCAM, MODEL_NAMES, load_model, preprocess

    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out_dir = HERE / "results" / "face_regions"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 정렬된 정면 얼굴의 대략적인 5점으로 영역 함수 자체 점검: 모든 픽셀이 영역 하나, 각 점이 제 영역 안
    toy = np.array([[80, 95], [144, 95], [112, 130], [88, 160], [136, 160]], float)
    lab = face_regions(toy)
    assert lab.max() < len(REGIONS) and set(np.unique(lab)) == set(range(len(REGIONS)))
    assert lab[95, 80] == 1 and lab[95, 144] == 2 and lab[130, 112] == 3 and lab[160, 112] == 4 and lab[3, 3] == OTHER
    assert abs(np.nansum(attention_share(np.random.rand(224, 224), lab)) - 100) < 1e-9

    val = pd.read_csv(HERE.parent / "splits_csv_v2" / "val.csv")
    val["src"] = np.where(val.path.str.contains("A"), "AAF", np.where(val.path.str.endswith(".chip.jpg"), "UTKFace", "FairFace"))
    picks = pd.concat([g.sample(args.n // 3, random_state=7) for _, g in val.groupby("src")])
    cams = {n: GradCAM(load_model(n, device=args.device), n) for n in MODEL_NAMES}

    def demo_photo(im, tilt=0):
        rng = np.random.default_rng(0)
        bg = Image.fromarray(rng.normal(128, 20, (960, 1280, 3)).clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3))
        bg.paste(im.resize((300, 300)).rotate(tilt, resample=Image.BILINEAR, fillcolor=(128, 128, 128)), (600, 250))
        return bg

    DOTS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    lm_err, rows, tiles, areas = [], [], [], []
    for i, (path, src) in enumerate(zip(picks.path, picks.src)):
        im = Image.open(args.image_root / path).convert("RGB")
        for align, tilt in [(False, 0), (True, 0), (True, 15)]:
            photo = demo_photo(im, tilt)
            calls, detect = [], fa.detect
            fa.detect = lambda rgb: calls.append(detect(rgb)) or calls[-1]
            face, _, lm = fa.crop_face(photo, align=align, return_landmarks=True)
            replay = lambda: (lambda it: lambda rgb: next(it))(iter(calls))
            if face is None:
                fa.detect = detect
                print(f"  {path} 정렬={align} 기울기={tilt}: 얼굴 못 찾음")
                continue
            fa.detect = replay()
            loc = fa.locate(np.asarray(photo), align)
            back = (np.c_[loc["lm"], np.ones(5)] @ np.linalg.inv(np.r_[loc["M"], [[0, 0, 1]]]).T)[:, :2] - loc["pad"]
            marked = photo.copy()
            dr = ImageDraw.Draw(marked)
            for (x, y), c in zip(back, DOTS):
                dr.ellipse((x - 4, y - 4, x + 4, y + 4), fill=c)
            fa.detect = replay()
            mface, _, mlm = fa.crop_face(marked, align=align, return_landmarks=True)
            fa.detect = detect
            assert np.allclose(mlm, lm)
            arr = np.asarray(mface).astype(int)
            for (x, y), c in zip(mlm, DOTS):
                yy, xx = np.nonzero(np.abs(arr - c).sum(-1) < 90)
                near = np.hypot(xx - x, yy - y) < 25
                e = np.hypot(xx[near].mean() - x, yy[near].mean() - y) if near.any() else np.inf
                lm_err.append((align, tilt, e))
            labels = face_regions(lm)
            tiles.append(draw_regions(face, labels, lm))
            if (align, tilt) != (fa.DEFAULT_ALIGN, 0):
                continue
            areas.append(100 * np.bincount(labels.ravel(), minlength=len(REGIONS)) / labels.size)
            for name in MODEL_NAMES:
                heat, cls, prob = cams[name](preprocess([face]))
                share = attention_share(heat[0], labels)
                assert abs(share.sum() - 100) < 1e-6, share.sum()
                p0 = float(prob[0, cls[0]])
                drop = occlusion(cams[name].model, face, labels, int(cls[0]), preprocess) - p0
                rows.append(dict(photo=f"{i + 1}({src})", model=name, band=BANDS[cls[0]], prob=round(100 * p0, 1),
                                 share_top=REGIONS[int(np.argmax(share))], share_top_pct=round(share.max(), 1),
                                 mouth_share=round(share[4], 1), occl_top=REGIONS[int(np.argmin(drop))],
                                 occl_top_pp=round(100 * float(drop.min()), 1),
                                 match=REGIONS[int(np.argmax(share))] == REGIONS[int(np.argmin(drop))]))

    err = pd.DataFrame(lm_err, columns=["align", "tilt", "err_px"])
    print("① 5점 좌표 오차 (반환한 224 좌표 vs 224 사진 속 색 점, px — 점 5개 × 사진):")
    print(err.groupby(["align", "tilt"]).err_px.agg(["mean", "median", "max", "count"]).round(2).to_string())
    assert np.isfinite(err.err_px).all() and err.err_px.max() < 2, "5점이 잘라낸 얼굴과 맞지 않음"
    W = 3
    sheet = Image.new("RGB", (448 * W, 448 * ((len(tiles) + W - 1) // W)), "white")
    for k, t in enumerate(tiles):
        sheet.paste(t, (448 * (k % W), 448 * (k // W)))
    sheet.save(out_dir / "regions_debug.jpg", quality=90)
    print(f"② 영역 경계 그림: {out_dir / 'regions_debug.jpg'} (줄마다 사진 1장: 정렬 끔 | 정렬 켬 | 15도 기울임+정렬 켬)")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "compare.csv", index=False, encoding="utf-8-sig")
    print("③ 주목 비중 합 = 100% (모든 사진·모델). 영역 면적 비율 평균(히트맵이 고르면 비중 = 면적):")
    print("   " + " · ".join(f"{r} {a:.0f}%" for r, a in zip(REGIONS, np.mean(areas, 0))))
    print("④ 주목 비중 1위 vs 가림 실험 하락 1위:")
    print(df.to_string(index=False))
    print(df.groupby("model").agg(일치=("match", "sum"), 사진=("match", "size"), 입비중평균=("mouth_share", "mean")).round(1).to_string())
