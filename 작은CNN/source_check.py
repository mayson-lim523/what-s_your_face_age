# #############################################################################
#
#   출처(데이터셋)별 자르기 차이가 지름길 학습인지 확인 (개발일지 17)
#   인공지능 4조 · 작은 CNN 담당: LIM
#
#   [하는 일] (--out-dir 폴더)
#     1 source_rule.csv       : 파일 이름 규칙별 장수·번호 범위 (출처 구분 규칙 점검)
#     2 band_source_count.csv : train·val 나이대 × 출처 장수와 비율
#     3 source_metrics.csv    : val 출처별 정확도·한 칸 정확도·macro-F1
#       band_source_recall.csv: val 나이대 × 출처별 재현율
#     4 face_width.csv/png    : 출처별 Haar 얼굴 상자 폭 ÷ 사진 폭 분포 (시연 app.detect_face와 같은 검출)
#     5 mask_compare.csv      : 가장자리 10% 가림 vs 같은 면적의 얼굴 중앙 가림 → 나이대별 재현율, 예측이 바뀐 비율 (출처별 포함)
#
#   [출처 규칙] 팀 전처리 코드(`설명 & python 코드/`, `개발일지/kimgyuwon/`) 기준
#     숫자A나이.jpg (00001A02.jpg) → AAF      (AAF.py: name.split("A")[1] 을 나이로 읽음)
#     나이_성별_인종_날짜.jpg.chip.jpg → UTKFace (UTK.py: filename.split("_")[0] 을 나이로 읽음)
#     숫자.jpg (1~86743)           → FairFace (fairface_train/val.py: CSV file 열 = train/숫자.jpg)
#     그 외                         → 기타     (v2에서는 0장 — AFAD 형식 없음)
#
#   [주의] 검증·학습 CSV만 사용 (test.csv 사용 금지). 재학습 없음
#   [실행] 프로젝트 폴더(사진·results가 있는 곳)에서
#     python <저장소>\작은CNN\source_check.py --weights results\small_cnn_v2_lr1e-3\small_cnn_best.pth
#            --csv-dir splits_csv_v2 --image-root division_224AFAD-X
#            --pred-csv results\small_cnn_v2_lr1e-3\eval_val\predictions.csv
#            --out-dir results\small_cnn_v2_lr1e-3\source --device cpu
#
# #############################################################################


# =============================================================================
#   ▶ [모듈 S-1 시작] 설정·옵션·출처 규칙
# =============================================================================

import argparse                                         # 명령줄 옵션
import os                                               # 경로
import re                                               # 파일 이름 규칙
import sys                                              # import 경로

import numpy as np                                      # 배열
import pandas as pd                                     # 표
import torch                                            # PyTorch
from PIL import Image                                   # 사진
from sklearn.metrics import f1_score                    # macro-F1
import matplotlib                                       # 그래프
matplotlib.use("Agg")                                   # 파일로만 저장
import matplotlib.pyplot as plt                         # 그림

HERE = os.path.dirname(os.path.abspath(__file__))       # 저장소/작은CNN
sys.path.insert(0, HERE)                                # 같은 폴더 코드
sys.path.insert(0, os.path.join(HERE, "..", "시연"))     # 시연/app.py (detect_face)
from gradcam_analysis import load_data, load_model, BAND_NAMES   # 검증 데이터·모델 (predictions.csv 순서 점검 포함)
from edge_check import mask_edge, MASK_FRAC, GRAY      # 가장자리 10% 가림 (개발일지 16과 같은 함수)
import app                                              # detect_face (시연과 같은 Haar 검출)

SOURCES = ["FairFace", "UTKFace", "AAF", "기타"]        # 출처 순서 (표 열 순서)
NUM_CLASSES = len(BAND_NAMES)                           # 7
WIDTH_PER_SOURCE = 600                                  # 4번: 출처별 Haar 측정 장수 (시드 고정)


def source_of(path):                                    # 파일 경로 → 출처 이름
    name = os.path.basename(path)                       # 파일 이름만
    if re.fullmatch(r"\d+A\d+\.jpg", name):             # 00001A02.jpg
        return "AAF"
    if name.endswith(".jpg.chip.jpg"):                  # 나이_성별_인종_날짜.jpg.chip.jpg
        return "UTKFace"
    if re.fullmatch(r"\d+\.jpg", name):                 # 숫자.jpg
        return "FairFace"
    return "기타"


def parse_args():                                       # 명령줄 옵션
    p = argparse.ArgumentParser(description="출처별 자르기 차이 · 지름길 학습 확인")
    p.add_argument("--weights", required=True)          # 가중치
    p.add_argument("--csv-dir", required=True)          # train.csv, val.csv 가 있는 폴더 (test.csv는 읽지 않음)
    p.add_argument("--image-root", required=True)       # 사진 기준 폴더
    p.add_argument("--pred-csv", required=True)         # val predictions.csv
    p.add_argument("--out-dir", required=True)          # 결과 폴더
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--parts", default="12345")          # 할 일 번호
    p.add_argument("--seed", type=int, default=0)       # 4번 표본 시드
    return p.parse_args()


def save(df, args, name, index=False):                  # CSV 저장 + 화면 출력
    df.to_csv(os.path.join(args.out_dir, name), index=index, encoding="utf-8-sig")
    print(f"\n[{name}]"); print(df.to_string(index=index))

# -----------------------------------------------------------------------------
# ◀ [모듈 S-1 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   ▶ [모듈 S-2 시작] 1·2번: 규칙 점검, 나이대 × 출처 장수
# =============================================================================

def part12(args):
    rule, counts = [], []                               # 규칙 점검 행, 장수 표들
    for split in ["train", "val"]:                      # test는 읽지 않음
        df = pd.read_csv(os.path.join(args.csv_dir, f"{split}.csv"))
        names = df["path"].str.split("/").str[-1]       # 파일 이름
        df["출처"] = df["path"].map(source_of)
        for s in SOURCES:                               # 출처별 이름 번호 범위 (규칙이 맞는지 확인용)
            n = names[df["출처"] == s]
            num = n.str.extract(r"^(\d+)")[0].dropna().astype(int)   # 맨 앞 숫자 (FairFace 번호, AAF 사람번호, UTK 나이)
            rule.append({"분할": split, "출처": s, "장수": len(n),
                         "앞숫자_최소": num.min() if len(num) else None, "앞숫자_최대": num.max() if len(num) else None,
                         "예": n.iloc[0] if len(n) else ""})
        t = pd.crosstab(df["band"].map(dict(enumerate(BAND_NAMES))), df["출처"]).reindex(BAND_NAMES)   # 나이대 × 출처 장수
        t = t.reindex(columns=[s for s in SOURCES if s in t.columns])
        pct = t.div(t.sum(axis=1), axis=0).mul(100).round(1).add_suffix("(%)")   # 나이대 안에서의 비율
        both = pd.concat([t, pct], axis=1)
        both.loc["전체"] = list(t.sum()) + list((t.sum() / t.values.sum() * 100).round(1))
        both.insert(0, "분할", split)
        counts.append(both)
    save(pd.DataFrame(rule), args, "source_rule.csv")
    save(pd.concat(counts), args, "band_source_count.csv", index=True)

# -----------------------------------------------------------------------------
# ◀ [모듈 S-2 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   ▶ [모듈 S-3 시작] 3번: val 출처별 성능
# =============================================================================

def part3(ds, y, p, args):
    src = np.array([source_of(x) for x in ds.paths])    # 사진별 출처
    rows = []
    for s in SOURCES:
        m = src == s
        if not m.any():
            continue
        present = sorted(set(y[m]))                     # 이 출처에 실제로 있는 나이대만 macro-F1에 넣음
        rows.append({"출처": s, "장수": int(m.sum()), "정확도": (y[m] == p[m]).mean(),
                     "한칸정확도": (abs(y[m] - p[m]) <= 1).mean(),
                     "macro-F1(있는 나이대)": f1_score(y[m], p[m], labels=present, average="macro", zero_division=0),
                     "나이대 수": len(present)})
    rows.append({"출처": "전체", "장수": len(y), "정확도": (y == p).mean(), "한칸정확도": (abs(y - p) <= 1).mean(),
                 "macro-F1(있는 나이대)": f1_score(y, p, average="macro"), "나이대 수": NUM_CLASSES})
    save(pd.DataFrame(rows).round(3), args, "source_metrics.csv")

    rec = []                                            # 나이대 × 출처 재현율 (장수 함께)
    for b in range(NUM_CLASSES):
        r = {"나이대": BAND_NAMES[b]}
        for s in SOURCES:
            m = (y == b) & (src == s)
            if m.sum():
                r[f"{s}_재현율"] = round((p[m] == b).mean(), 3)
                r[f"{s}_장수"] = int(m.sum())
                r[f"{s}_평균예측차"] = round((p[m] - b).mean(), 2)   # 예측 − 정답 평균 (+면 나이 많게)
        rec.append(r)
    save(pd.DataFrame(rec), args, "band_source_recall.csv")
    return src

# -----------------------------------------------------------------------------
# ◀ [모듈 S-3 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   ▶ [모듈 S-4 시작] 4번: 출처별 얼굴 크기 (Haar 상자 폭 ÷ 사진 폭)
# =============================================================================

def part4(ds, src, args):
    rng = np.random.default_rng(args.seed)              # 시드 고정
    rows = []
    for s in SOURCES:
        idx = np.where(src == s)[0]
        if len(idx) == 0:
            continue
        idx = rng.choice(idx, min(WIDTH_PER_SOURCE, len(idx)), replace=False)   # 출처별 최대 600장
        for i in idx:
            img = Image.open(ds.paths[i])
            box = app.detect_face(img)                  # (x, y, w, h) 또는 None
            rows.append({"출처": s, "검출": box is not None,
                         "상자폭비": box[2] / img.size[0] if box else np.nan,
                         "중심y": (box[1] + box[3] / 2) / img.size[1] if box else np.nan})
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(args.out_dir, "face_width_per_image.csv"), index=False, encoding="utf-8-sig")
    g = d.groupby("출처", sort=False)
    summ = pd.DataFrame({"표본": g.size(), "검출률": g["검출"].mean().round(3)})
    q = g["상자폭비"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack().round(3)
    q.columns = [f"폭비_{int(c * 100)}%" for c in q.columns]
    summ = summ.join(q).join(g["중심y"].median().round(3).rename("중심y_중앙값"))
    save(summ.reset_index(), args, "face_width.csv")

    fig, ax = plt.subplots(figsize=(8, 4))              # 출처별 분포 겹쳐 그리기
    for s in summ.index:
        v = d.loc[(d["출처"] == s) & d["검출"], "상자폭비"]
        ax.hist(v, bins=np.arange(0.3, 1.45, 0.025), histtype="step", lw=2, density=True,
                label=f"{s} (검출 {len(v)}/{(d['출처'] == s).sum()})")
    ax.set_xlabel("Haar 얼굴 상자 폭 ÷ 사진 폭 (1 이상 = 얼굴이 사진보다 넓게 잡힘)")
    ax.set_ylabel("비율 (밀도)")
    ax.set_title("출처별 얼굴 크기 분포 (검증 사진, 출처별 최대 600장)")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "face_width.png"), dpi=120); plt.close(fig)

# -----------------------------------------------------------------------------
# ◀ [모듈 S-4 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   ▶ [모듈 S-5 시작] 5번: 가장자리 가림 vs 같은 면적 중앙 가림
# =============================================================================

def mask_center(x):                                     # 가장자리 10% 가림과 같은 면적의 가운데 정사각형을 회색으로
    m = int(round(224 * MASK_FRAC))                     # 가장자리 두께 22
    area = 224 * 224 - (224 - 2 * m) ** 2               # 가장자리 가림 면적 = 17,776
    side = int(round(area ** 0.5))                      # 같은 면적 정사각형 한 변 = 133 (17,689, 0.5% 차이)
    a = (224 - side) // 2                               # 시작 위치 (가운데)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    g = ((GRAY / 255 - mean) / std).to(x.device)        # 회색(128) 정규화 값
    x = x.clone()
    x[:, :, a:a + side, a:a + side] = g.expand(x.size(0), 3, side, side)
    return x


def part5(model, ds, y, src, args):
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)   # 순서 고정
    out = {"원본": [], "가장자리": [], "중앙": []}
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(args.device)
            out["원본"].append(model(x).argmax(1).cpu())
            out["가장자리"].append(model(mask_edge(x)).argmax(1).cpu())
            out["중앙"].append(model(mask_center(x)).argmax(1).cpu())
    o, e, c = (torch.cat(v).numpy() for v in out.values())
    rows = []
    for name, m in [(BAND_NAMES[b], y == b) for b in range(NUM_CLASSES)] + \
                   [(f"[출처] {s}", src == s) for s in SOURCES if (src == s).any()] + [("전체", y >= 0)]:
        rows.append({"묶음": name, "장수": int(m.sum()),
                     "정답률_원본": (o[m] == y[m]).mean(),             # 나이대 묶음 = 재현율, 출처·전체 묶음 = 정확도
                     "정답률_가장자리가림": (e[m] == y[m]).mean(),
                     "정답률_중앙가림": (c[m] == y[m]).mean(),
                     "바뀐비율_가장자리": (e[m] != o[m]).mean(),        # 원본 예측과 달라진 비율
                     "바뀐비율_중앙": (c[m] != o[m]).mean(),
                     "평균예측차_가장자리": (e[m] - o[m]).mean(),       # 가림 후 예측 − 원본 예측 (+면 나이 많게 바뀜)
                     "평균예측차_중앙": (c[m] - o[m]).mean()})
    save(pd.DataFrame(rows).round(3), args, "mask_compare.csv")
    side = int(round((224 * 224 - (224 - 2 * int(round(224 * MASK_FRAC))) ** 2) ** 0.5))
    print(f"가린 면적: 가장자리 {224 * 224 - 180 * 180}px, 중앙 {side}×{side} = {side * side}px")

# -----------------------------------------------------------------------------
# ◀ [모듈 S-5 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   실행부
# =============================================================================
if __name__ == "__main__":
    args = parse_args()
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
    os.makedirs(args.out_dir, exist_ok=True)
    if any(k in args.parts for k in "12"):
        part12(args)
    if any(k in args.parts for k in "345"):
        args.csv = os.path.join(args.csv_dir, "val.csv")   # load_data가 쓰는 이름 (검증만)
        ds, y, p = load_data(args)                      # val + predictions.csv (순서 점검)
        src = part3(ds, y, p, args)
        if "4" in args.parts:
            part4(ds, src, args)
        if "5" in args.parts:
            part5(load_model(args), ds, y, src, args)
    print("결과:", args.out_dir)

# #############################################################################
# 모듈 끝: source_check.py
# #############################################################################
