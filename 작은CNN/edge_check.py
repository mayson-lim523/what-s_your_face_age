# #############################################################################
#
#   20·30대 가장자리 Grad-CAM 원인 확인 (개발일지 15 "다음 할 일")
#   인공지능 4조 · 작은 CNN 담당: LIM
#
#   [하는 일] (--out-dir 폴더)
#     1 edge_per_image.csv : 20·30대 맞힌 사진 전체의 사진별 "얼굴 밖" 비율 + 사진 특징(테두리 밝기, 얼굴 폭, 출처 형식)
#     2 edge_top_bottom.png: 얼굴 밖 비율 상위 20장 | 하위 20장 원본 격자
#     3 edge_groups.csv    : 두 그룹의 특징 비교 표
#     4 mask_recall.csv    : 검증 사진 가장자리 10%를 회색으로 가리고 다시 예측 → 나이대별 재현율 전후 비교 (--parts 에 4가 있을 때)
#
#   [재사용] 시연/gradcam.py (GradCAM, target_layer), gradcam_analysis.py (load_data, load_model, region_masks)
#   [주의] 검증 데이터만 사용 (test.csv 사용 금지). 학습은 다시 하지 않음
#
#   [실행] 프로젝트 폴더(사진·results가 있는 곳)에서
#     python <저장소>\작은CNN\edge_check.py --weights results\small_cnn_v2_lr1e-3\small_cnn_best.pth
#            --csv splits_csv_v2\val.csv --image-root division_224AFAD-X
#            --pred-csv results\small_cnn_v2_lr1e-3\eval_val\predictions.csv
#            --out-dir results\small_cnn_v2_lr1e-3\gradcam\edge --device cpu
#
# #############################################################################


# =============================================================================
#   ▶ [모듈 E-1 시작] 설정·옵션
# =============================================================================

import argparse                                         # 명령줄 옵션
import os                                               # 경로
import re                                               # 파일 이름 형식 구분
import sys                                              # import 경로

import cv2                                              # Haar 얼굴 검출 (얼굴 폭 측정용)
import numpy as np                                      # 배열 계산
import pandas as pd                                     # 표
import torch                                            # PyTorch
from PIL import Image                                   # 사진 열기
import matplotlib                                       # 그래프 설정
matplotlib.use("Agg")                                   # 파일로만 저장
import matplotlib.pyplot as plt                         # 그림

HERE = os.path.dirname(os.path.abspath(__file__))       # 저장소/작은CNN
sys.path.insert(0, HERE)                                # 같은 폴더 코드 import
sys.path.insert(0, os.path.join(HERE, "..", "시연"))     # 시연/gradcam.py
from gradcam import GradCAM, target_layer               # Grad-CAM (복사 안 함)
from gradcam_analysis import load_data, load_model, region_masks, BAND_NAMES   # 개발일지 15 코드 재사용

BANDS = [2, 3]                                          # 20대, 30대
TOP_K = 20                                              # 상위·하위 장수
MASK_FRAC = 0.10                                        # 4번: 가릴 가장자리 비율 (한 변의 10% = 22픽셀)
GRAY = 128                                              # 가리는 회색 (0~255)
BORDER = 8                                              # 테두리 밝기를 잴 가장자리 두께 (픽셀)


def parse_args():                                       # 명령줄 옵션
    p = argparse.ArgumentParser(description="20·30대 가장자리 Grad-CAM 원인 확인")
    p.add_argument("--weights", required=True)          # 가중치
    p.add_argument("--csv", required=True)              # 검증 CSV (test.csv 금지)
    p.add_argument("--image-root", required=True)       # 사진 기준 폴더
    p.add_argument("--pred-csv", required=True)         # predictions.csv
    p.add_argument("--out-dir", required=True)          # 결과 폴더
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")   # 장치
    p.add_argument("--parts", default="1234")           # 할 일 번호
    return p.parse_args()

# -----------------------------------------------------------------------------
# ◀ [모듈 E-1 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   ▶ [모듈 E-2 시작] 사진 특징 재기
# =============================================================================

def source_of(path):                                    # 파일 이름 형식 → 출처 추정 이름
    name = os.path.basename(path)                       # 파일 이름만
    if name.endswith(".chip.jpg"):                      # 나이_성별_인종_날짜.jpg.chip.jpg
        return "UTKFace형식"
    if re.fullmatch(r"\d+A\d+\.jpg", name):             # 00001A02.jpg (사람번호 A 나이). FG-NET도 같은 형식이지만 FG-NET은 1,002장뿐이라 출처를 단정하지 않음
        return "숫자A나이형식"
    if re.fullmatch(r"\d{5}\.jpg", name):               # 5자리 숫자
        return "숫자5자리"
    return "숫자4자리이하"                                # 그 외 숫자 이름


FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")   # OpenCV 기본 얼굴 검출기


def features(path):                                     # 사진 한 장 → 특징 딕셔너리
    img = np.asarray(Image.open(path).convert("RGB").resize((224, 224)), dtype=np.float32)   # (224, 224, 3)
    gray = img.mean(axis=2)                             # 밝기
    edge = np.concatenate([gray[:BORDER].ravel(), gray[-BORDER:].ravel(),            # 위·아래 띠
                           gray[:, :BORDER].ravel(), gray[:, -BORDER:].ravel()])    # 왼쪽·오른쪽 띠
    faces = FACE_CASCADE.detectMultiScale(np.uint8(gray), scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))   # 얼굴 상자들
    w = max((f[2] for f in faces), default=np.nan)      # 가장 큰 얼굴 폭 (없으면 NaN)
    return {"테두리_밝기": edge.mean(),                  # 0~255
            "검은테두리": float((edge < 15).mean() > 0.5),   # 가장자리 절반 이상이 거의 검정 → 1
            "테두리_표준편차": edge.std(),                 # 낮으면 단색 배경·여백
            "얼굴폭_px": w,                              # Haar 얼굴 폭 (224 기준)
            "출처형식": source_of(path)}

# -----------------------------------------------------------------------------
# ◀ [모듈 E-2 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   ▶ [모듈 E-3 시작] 1~3번: 사진별 얼굴 밖 비율, 상위·하위 격자, 그룹 비교
# =============================================================================

def part123(model, ds, y, p, args):
    cam = GradCAM(model, *target_layer(model, "small_cnn"))   # hook
    outside = region_masks()[4]                         # 얼굴 밖 마스크 (224, 224)
    rows = []                                           # 사진별 결과
    idx = np.where(np.isin(y, BANDS) & (y == p))[0]     # 20·30대 맞힌 사진 전체
    for n, i in enumerate(idx):
        heat, _ = cam(ds[i][0].unsqueeze(0).to(args.device), class_idx=int(y[i]))   # 정답 나이대 히트맵
        r = (heat * outside).sum() / (heat.sum() + 1e-8)   # 얼굴 밖 비율
        rows.append({"번호": i, "path": ds.paths[i], "나이대": BAND_NAMES[y[i]], "얼굴밖(%)": 100 * r, **features(ds.paths[i])})
        if n % 300 == 0:
            print(f"  {n}/{len(idx)}")
    cam.remove()
    df = pd.DataFrame(rows).sort_values("얼굴밖(%)", ascending=False)   # 얼굴 밖 비율 높은 순
    df.round(2).to_csv(os.path.join(args.out_dir, "edge_per_image.csv"), index=False, encoding="utf-8-sig")

    top, bot = df.head(TOP_K), df.tail(TOP_K)           # 상위 20, 하위 20
    fig, axes = plt.subplots(4, 10, figsize=(16, 7.6))    # 위 2줄 = 상위 20, 아래 2줄 = 하위 20
    for k, (_, r) in enumerate(pd.concat([top, bot]).iterrows()):
        ax = axes.flat[k]
        ax.imshow(Image.open(r["path"]).convert("RGB").resize((224, 224)))
        ax.set_title(f"{r['나이대']} {r['얼굴밖(%)']:.0f}%\n{os.path.basename(r['path'])[:18]}", fontsize=7,
                     color="red" if k < TOP_K else "blue")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"20·30대 맞힌 사진 {len(df)}장 중 Grad-CAM '얼굴 밖' 비율 상위 {TOP_K}장(빨강, 위 2줄) | 하위 {TOP_K}장(파랑, 아래 2줄)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(os.path.join(args.out_dir, "edge_top_bottom.png"), dpi=100); plt.close(fig)

    groups = {"상위20": top, "하위20": bot, "전체": df}   # 비교할 그룹
    comp = []
    for g, d in groups.items():
        row = {"그룹": g, "장수": len(d), "얼굴밖(%)": d["얼굴밖(%)"].mean(),
               "테두리_밝기": d["테두리_밝기"].mean(), "검은테두리_장수": int(d["검은테두리"].sum()),
               "테두리_표준편차": d["테두리_표준편차"].mean(),
               "얼굴검출_장수": int(d["얼굴폭_px"].notna().sum()), "얼굴폭_px(평균)": d["얼굴폭_px"].mean(),
               "20대_장수": int((d["나이대"] == BAND_NAMES[2]).sum())}
        for s in ["숫자5자리", "UTKFace형식", "숫자A나이형식", "숫자4자리이하"]:   # 출처 형식별 장수
            row[s] = int((d["출처형식"] == s).sum())
        comp.append(row)
    comp = pd.DataFrame(comp).round(1)
    comp.to_csv(os.path.join(args.out_dir, "edge_groups.csv"), index=False, encoding="utf-8-sig")
    print(comp.to_string(index=False))
    # 얼굴 밖 비율을 4등분 했을 때 출처 형식 비율 (상위·하위 20장보다 큰 표본으로 확인)
    df["사분위"] = pd.qcut(df["얼굴밖(%)"], 4, labels=["Q1(낮음)", "Q2", "Q3", "Q4(높음)"])
    q = pd.crosstab(df["사분위"], df["출처형식"], normalize="index").mul(100).round(1)
    q.to_csv(os.path.join(args.out_dir, "edge_quartile_source.csv"), encoding="utf-8-sig")
    print(q.to_string())

# -----------------------------------------------------------------------------
# ◀ [모듈 E-3 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   ▶ [모듈 E-4 시작] 4번: 가장자리 10% 회색으로 가리고 다시 예측
# =============================================================================

def mask_edge(x):                                       # x: (B, 3, 224, 224) 정규화된 텐서 → 가장자리를 회색으로
    m = int(round(224 * MASK_FRAC))                     # 22픽셀
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)   # ImageNet 평균
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)    # ImageNet 표준편차
    g = ((GRAY / 255 - mean) / std).to(x.device)        # 회색(128)을 정규화한 값 (채널별)
    x = x.clone()
    for sl in [(slice(None), slice(None), slice(0, m)), (slice(None), slice(None), slice(224 - m, 224)),   # 위, 아래
               (slice(None), slice(None), slice(None), slice(0, m)), (slice(None), slice(None), slice(None), slice(224 - m, 224))]:   # 왼쪽, 오른쪽
        x[sl] = g.expand_as(x[sl])
    return x


def part4(model, ds, y, args):
    preds = {"원본": [], "가장자리 가림": []}
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)   # 순서 고정
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(args.device)
            preds["원본"].append(model(x).argmax(1).cpu())
            preds["가장자리 가림"].append(model(mask_edge(x)).argmax(1).cpu())
    rows = []
    pa, pb = (torch.cat(v).numpy() for v in preds.values())
    for b in range(len(BAND_NAMES)):
        t = y == b
        rows.append({"나이대": BAND_NAMES[b], "장수": int(t.sum()),
                     "재현율_원본": (pa[t] == b).mean(), "재현율_가림": (pb[t] == b).mean(),
                     "예측장수_원본": int((pa == b).sum()), "예측장수_가림": int((pb == b).sum())})
    df = pd.DataFrame(rows)
    df["재현율_차이"] = df["재현율_가림"] - df["재현율_원본"]
    df.loc[len(df)] = {"나이대": "전체 정확도", "장수": len(y), "재현율_원본": (pa == y).mean(), "재현율_가림": (pb == y).mean(),
                       "재현율_차이": (pb == y).mean() - (pa == y).mean()}
    df = df.round(4)
    df.to_csv(os.path.join(args.out_dir, "mask_recall.csv"), index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"예측이 바뀐 사진: {(pa != pb).sum()}장 / {len(y)}장")

# -----------------------------------------------------------------------------
# ◀ [모듈 E-4 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   실행부
# =============================================================================
if __name__ == "__main__":
    args = parse_args()
    assert "test" not in os.path.basename(args.csv).lower(), "시험 데이터는 쓰지 않습니다"
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False   # 한글 글꼴
    os.makedirs(args.out_dir, exist_ok=True)
    ds, y, p = load_data(args)                          # 검증 데이터, 정답, 예측 (predictions.csv와 순서 점검 포함)
    model = load_model(args)                            # eval 상태 모델
    if any(c in args.parts for c in "123"):
        part123(model, ds, y, p, args)
    if "4" in args.parts:
        part4(model, ds, y, args)
    print("결과:", args.out_dir)

# #############################################################################
# 모듈 끝: edge_check.py
# #############################################################################
