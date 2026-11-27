# #############################################################################
#
#   작은 CNN(4층) Grad-CAM 판단근거 분석 — 새 데이터 v2 가중치
#   인공지능 4조 · 작은 CNN 담당: LIM
#
#   [만드는 그림·표] (--out-dir 폴더)
#     1 correct_by_band.png : 나이대별 맞힌 사진 --per-band장 (확률 높은 순) — 원본 | 히트맵 | 겹친 그림
#     2 confusion_pairs.png : 자주 헷갈린 쌍의 틀린 사진 (틀린 확률 높은 순) — 같은 3칸 형식
#     3 mean_cam_by_band.png: 나이대별 맞힌 사진(최대 500장) Grad-CAM 평균 → 평균 얼굴 위에 구역선과 함께
#     4 region_ratio.csv/png: 이마 / 눈 / 코·볼 / 입·턱 / 얼굴 밖 5구역의 히트맵 에너지 비율 (나이대별 · 맞음/틀림별)
#
#   [재사용] Grad-CAM: 시연/gradcam.py (GradCAM, target_layer, overlay)
#            모델·데이터: CNN_4layers_unified.py (SmallCNN, FaceCSVDataset, val_transform)
#            나이대 이름: eval_small_cnn.py (BAND_NAMES)
#   [주의] 검증 데이터만 사용 (test.csv 사용 금지). 사진은 이미 얼굴 정렬·224×224라 얼굴 검출 안 함
#
#   [실행] 프로젝트 폴더(사진·results가 있는 곳)에서
#     python <저장소>\작은CNN\gradcam_analysis.py --weights results\small_cnn_v2_lr1e-3\small_cnn_best.pth
#            --csv splits_csv_v2\val.csv --image-root division_224AFAD-X
#            --pred-csv results\small_cnn_v2_lr1e-3\eval_val\predictions.csv --out-dir results\small_cnn_v2_lr1e-3\gradcam
#     일부만: --parts 1 (1번 그림만) / --parts 234
#
# #############################################################################


# =============================================================================
#   ▶ [모듈 G-1 시작] 설정·명령줄 옵션
#   출력: 상수들, parse_args()
# =============================================================================

import argparse                                         # 명령줄 옵션 읽기
import os                                               # 경로 합치기, 폴더 만들기
import sys                                              # import 경로(sys.path)에 시연 폴더 추가

import numpy as np                                      # 배열 계산 (평균 히트맵, 구역 비율)
import pandas as pd                                     # CSV 읽기·쓰기
import torch                                            # PyTorch 본체
from PIL import Image                                   # 사진 열기
import matplotlib                                       # 그래프 설정용
matplotlib.use("Agg")                                   # 창 없이 파일로만 저장
import matplotlib.pyplot as plt                         # 그림 그리기
import matplotlib.cm as mcm                             # 히트맵 컬러맵 (jet)

HERE = os.path.dirname(os.path.abspath(__file__))       # 이 파일이 있는 폴더 (저장소/작은CNN)
sys.path.insert(0, HERE)                                # 같은 폴더의 학습·평가 코드 import용
sys.path.insert(0, os.path.join(HERE, "..", "시연"))     # 시연/gradcam.py import용 (복사하지 않고 그대로 사용)
from gradcam import GradCAM, overlay, target_layer      # Grad-CAM 직접 구현 모듈 (대상: SmallCNN conv4 블록 출력 = gap 입력)
from CNN_4layers_unified import SmallCNN, FaceCSVDataset, val_transform   # 모델 구조, CSV 데이터셋, 검증 전처리 (학습 때와 같음)
from eval_small_cnn import BAND_NAMES                   # ["0-9세", "10대", ..., "60세 이상"]

NUM_CLASSES = 7                                         # 나이대 7개
# 2번 그림: 미리 정한 헷갈린 쌍 (정답 번호, 예측 번호) — 20대→10대, 30대→20대, 40대→50대, 50대→60세 이상
PAIRS = [(2, 1), (3, 2), (4, 5), (5, 6)]
MAX_MEAN = 500                                          # 3·4번: 나이대(·맞음/틀림)당 최대 사진 수

# 4번: 얼굴 구역 경계 (224×224 픽셀 좌표). 평균 얼굴 그림(mean_cam_by_band.png)의 구역선으로 맞는지 확인할 것
FACE_X = (40, 184)                                      # (왼쪽, 오른쪽) 얼굴 안쪽 가로 범위 — 밖은 "얼굴 밖"
FACE_Y = (20, 214)                                      # (위, 아래) 얼굴 안쪽 세로 범위 — 이마 위 머리카락·턱 아래는 "얼굴 밖"
# 경계 근거: 검증 사진 2000장 평균 얼굴의 줄별 밝기 — 눈썹·눈 어두운 띠 y≈44~84 (눈 중심 ≈64), 콧구멍 ≈124, 입 ≈152
# (처음 값 75/115/155는 눈이 '이마' 구역에 걸쳐 들어가서 고침)
Y_EYES = 44                                             # 이마 | 눈 경계 (이 줄부터 눈썹·눈)
Y_NOSE = 84                                             # 눈 | 코·볼 경계
Y_MOUTH = 140                                           # 코·볼 | 입·턱 경계 (입 윗선 바로 위)
REGIONS = ["이마", "눈", "코·볼", "입·턱", "얼굴 밖"]       # 구역 이름 (표·그림 열 순서)


def parse_args():                                       # 명령줄 옵션을 읽어서 돌려줌
    p = argparse.ArgumentParser(description="SmallCNN Grad-CAM 판단근거 분석")   # (description=도움말 첫 줄)
    p.add_argument("--weights", required=True)          # 학습한 가중치 (.pth)
    p.add_argument("--csv", required=True)              # 검증 CSV (열: path, band, ...) — test.csv 쓰지 말 것
    p.add_argument("--image-root", required=True)       # CSV path의 기준 폴더
    p.add_argument("--pred-csv", required=True)         # eval_small_cnn.py가 만든 predictions.csv (--csv와 같은 순서)
    p.add_argument("--out-dir", required=True)          # 결과 저장 폴더
    p.add_argument("--per-band", type=int, default=6)   # 1·2번 그림에서 나이대(쌍)마다 보여줄 장수
    p.add_argument("--seed", type=int, default=42)      # 3·4번에서 500장을 뽑을 때 쓰는 시드
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")   # 장치 (GPU가 바쁘면 --device cpu)
    p.add_argument("--parts", default="1234")           # 만들 결과 번호 (예: 1 = 1번 그림만)
    return p.parse_args()                               # args.weights, args.image_root ... 로 사용

# -----------------------------------------------------------------------------
# ◀ [모듈 G-1 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   ▶ [모듈 G-2 시작] 데이터·모델 준비
#   출력: load_data(), load_model()
# =============================================================================

def load_data(args):                                    # → (FaceCSVDataset, 정답 배열, 예측 배열)
    ds = FaceCSVDataset(args.csv, args.image_root, val_transform)   # (CSV, 기준 폴더, 검증 전처리) — 증강 없음
    pred = pd.read_csv(args.pred_csv)                   # 사진별 정답·예측 (eval_small_cnn.py 출력)
    csv_paths = pd.read_csv(args.csv)["path"]           # CSV의 상대 경로
    assert len(pred) == len(ds), f"장수가 다름: predictions {len(pred)} vs csv {len(ds)}"   # 같은 검증 세트인지
    same = [p.replace("\\", "/").endswith(c) for p, c in zip(pred["path"], csv_paths)]   # 같은 순서인지 (경로 끝부분 비교)
    assert all(same), "predictions.csv와 --csv의 사진 순서가 다릅니다"
    y = np.array(ds.labels)                             # 정답 번호 (CSV band)
    assert (pred["band"].values == y).all(), "정답 번호가 다릅니다"   # 라벨도 일치하는지
    return ds, y, pred["pred"].values                   # (데이터셋, 정답, 예측)


def load_model(args):                                   # → eval 상태 SmallCNN
    model = SmallCNN().to(args.device)                  # 빈 모델 → 장치로
    model.load_state_dict(torch.load(args.weights, map_location=args.device))   # 가중치 채우기
    return model.eval()                                 # 평가 모드 (Dropout 끔, BN 고정) — Grad-CAM에 필요


def probs_of(model, ds, idx, cls, device):              # (idx=사진 번호들, cls=확률을 볼 나이대 번호들) → 확률 배열
    out = []                                            # 확률 모을 리스트
    with torch.no_grad():                               # 확률만 필요하므로 기울기 끔
        for s in range(0, len(idx), 128):               # 128장씩 묶어서
            x = torch.stack([ds[i][0] for i in idx[s:s + 128]]).to(device)   # (B, 3, 224, 224)
            p = torch.softmax(model(x).float(), dim=1).cpu().numpy()          # (B, 7) 확률
            out.append(p[np.arange(len(p)), cls[s:s + 128]])                  # 각 사진의 지정 나이대 확률
    return np.concatenate(out) if out else np.array([])  # (len(idx),)

# -----------------------------------------------------------------------------
# ◀ [모듈 G-2 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   ▶ [모듈 G-3 시작] 3칸 격자 그림 (원본 | 히트맵 | 겹친 그림) — 1·2번 공통
# =============================================================================

def raw_face(ds, i):                                    # i번째 사진 원본 (224×224 PIL, 모델이 본 크기)
    return Image.open(ds.paths[i]).convert("RGB").resize((224, 224))


def grid_figure(rows, cam, ds, path, title):            # (rows=[(행 이름, [(번호, 설명)...])...], cam=GradCAM, ds, path=저장, title=제목)
    ncol = max(len(r[1]) for r in rows) * 3             # 사진 1장당 3칸
    fig, axes = plt.subplots(len(rows), ncol, figsize=(ncol * 1.15, len(rows) * 1.45), squeeze=False)   # 격자
    for r, (name, items) in enumerate(rows):            # 행 = 나이대(또는 헷갈린 쌍)
        for k, ax in enumerate(axes[r]):                # 이 행의 모든 칸 먼저 축 끄기
            ax.axis("off")
        for j, (i, label, cls) in enumerate(items):     # (사진 번호, 설명, Grad-CAM 나이대)
            face = raw_face(ds, i)                      # 원본
            x = ds[i][0].unsqueeze(0).to(next(cam.model.parameters()).device)   # (1, 3, 224, 224)
            heat, _ = cam(x, class_idx=int(cls))        # 지정 나이대 히트맵 (맞힌 사진=정답, 틀린 사진=예측)
            for c, img in enumerate([face, heat, overlay(face, heat)]):   # 3칸
                ax = axes[r][j * 3 + c]                 # 칸 위치
                ax.imshow(img, cmap="jet" if c == 1 else None, vmin=0, vmax=1)   # 히트맵 칸은 jet, 0~1
                if c == 0:                              # 원본 칸 위에 설명
                    ax.set_title(label, fontsize=7)
        axes[r][0].text(-0.15, 0.5, name, transform=axes[r][0].transAxes,   # 행 이름 (왼쪽 바깥)
                        ha="right", va="center", fontsize=9)
    fig.suptitle(title, fontsize=11)                    # 전체 제목
    fig.tight_layout()                                  # 여백 조정
    fig.savefig(path, dpi=110, bbox_inches="tight")     # 저장 (행 이름 안 잘리게)
    plt.close(fig)                                      # 닫기

# -----------------------------------------------------------------------------
# ◀ [모듈 G-3 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   ▶ [모듈 G-4 시작] 1번 — 나이대별 맞힌 사진 (확률 높은 순)
# =============================================================================

def part1(model, cam, ds, y, p, args):
    rows = []                                           # 격자 행들
    for b in range(NUM_CLASSES):                        # 나이대마다
        idx = np.where((y == b) & (p == b))[0]          # 맞힌 사진 번호
        pr = probs_of(model, ds, idx, np.full(len(idx), b), args.device)   # 정답 나이대 확률
        top = idx[np.argsort(-pr)[:args.per_band]]      # 확률 높은 순 --per-band장
        top_p = np.sort(pr)[::-1][:args.per_band]       # 그 확률
        rows.append((f"{BAND_NAMES[b]}\n(맞힘 {len(idx)})", [(i, f"{q:.2f}", b) for i, q in zip(top, top_p)]))
    grid_figure(rows, cam, ds, os.path.join(args.out_dir, "correct_by_band.png"),
                "나이대별 맞힌 사진 (확률 높은 순) — 원본 | Grad-CAM(정답 나이대) | 겹친 그림")

# -----------------------------------------------------------------------------
# ◀ [모듈 G-4 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   ▶ [모듈 G-5 시작] 2번 — 자주 헷갈린 쌍의 틀린 사진
# =============================================================================

def part2(model, cam, ds, y, p, args):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)   # 혼동 행렬 (행=정답, 열=예측)
    np.add.at(cm, (y, p), 1)                            # 개수 세기
    off = [(cm[t, q], t, q) for t in range(NUM_CLASSES) for q in range(NUM_CLASSES) if t != q]   # 틀린 칸만
    top = sorted(off, reverse=True)[:len(PAIRS)]        # 개수 많은 순 상위 (미리 정한 쌍 수만큼)
    print("혼동 상위 쌍:", [f"{BAND_NAMES[t]}→{BAND_NAMES[q]} {n}장" for n, t, q in top])
    auto = {(t, q) for _, t, q in top}                  # 자동으로 뽑은 쌍
    if auto != set(PAIRS):                              # 미리 정한 목록과 다르면 알림
        print("⚠ 미리 정한 쌍과 다름 — 미리 정한 쌍:", [f"{BAND_NAMES[t]}→{BAND_NAMES[q]} {cm[t, q]}장" for t, q in PAIRS])
    rows = []                                           # 격자 행들
    for t, q in PAIRS:                                  # 미리 정한 쌍 그대로 그림
        idx = np.where((y == t) & (p == q))[0]          # 정답 t를 q로 예측한 사진
        pr = probs_of(model, ds, idx, np.full(len(idx), q), args.device)   # 틀린 나이대 확률
        top_i = idx[np.argsort(-pr)[:args.per_band]]    # 틀린 확률 높은 순 (자신 있게 틀린 사진)
        top_p = np.sort(pr)[::-1][:args.per_band]
        rows.append((f"{BAND_NAMES[t]}→{BAND_NAMES[q]}\n({len(idx)}장)", [(i, f"{v:.2f}", q) for i, v in zip(top_i, top_p)]))
    grid_figure(rows, cam, ds, os.path.join(args.out_dir, "confusion_pairs.png"),
                "자주 헷갈린 쌍의 틀린 사진 (틀린 확률 높은 순) — 원본 | Grad-CAM(예측 나이대) | 겹친 그림")

# -----------------------------------------------------------------------------
# ◀ [모듈 G-5 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   ▶ [모듈 G-6 시작] 3·4번 — 평균 히트맵, 구역 비율
# =============================================================================

def region_masks():                                     # → (5, 224, 224) 구역별 True/False
    yy, xx = np.mgrid[0:224, 0:224]                     # 픽셀 좌표 (행, 열)
    inside = (xx >= FACE_X[0]) & (xx < FACE_X[1]) & (yy >= FACE_Y[0]) & (yy < FACE_Y[1])   # 얼굴 안쪽 상자
    m = [inside & (yy < Y_EYES),                        # 이마
         inside & (yy >= Y_EYES) & (yy < Y_NOSE),       # 눈
         inside & (yy >= Y_NOSE) & (yy < Y_MOUTH),      # 코·볼
         inside & (yy >= Y_MOUTH),                      # 입·턱
         ~inside]                                       # 얼굴 밖 (배경·머리카락)
    return np.stack(m)                                  # 5구역이 겹치지 않고 전체를 덮음


def cam_stats(cam, ds, idx, cls, masks, device):        # (idx=사진 번호들, cls=Grad-CAM 나이대들) → (평균 히트맵, 평균 얼굴, 사진별 구역 비율)
    s_cam = np.zeros((224, 224))                        # 히트맵 합
    s_face = np.zeros((224, 224, 3))                    # 원본 얼굴 합
    ratios = []                                         # 사진별 5구역 비율
    for i, c in zip(idx, cls):                          # 사진마다 (Grad-CAM은 한 장씩)
        heat, _ = cam(ds[i][0].unsqueeze(0).to(device), class_idx=int(c))   # (224, 224) 0~1
        s_cam += heat                                   # 합에 더하기
        s_face += np.asarray(raw_face(ds, i), dtype=np.float64)   # 평균 얼굴용
        e = (masks * heat).sum(axis=(1, 2))             # 구역별 에너지 (히트맵 값 합)
        ratios.append(e / (e.sum() + 1e-8))             # 비율 (합 1)
    n = max(len(idx), 1)                                # 0 나눗셈 방지
    return s_cam / n, s_face / n, np.array(ratios).reshape(-1, len(REGIONS))


def sample(idx, k, rng):                                # 최대 k장 무작위 (시드 고정)
    return idx if len(idx) <= k else np.sort(rng.choice(idx, k, replace=False))


def draw_regions(ax):                                   # 평균 얼굴 위에 구역선 그리기
    x0, x1 = FACE_X; y0, y1 = FACE_Y                    # 얼굴 상자
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="white", lw=1))   # 얼굴 상자
    for yl in (Y_EYES, Y_NOSE, Y_MOUTH):                # 가로 경계선
        ax.plot([x0, x1], [yl, yl], color="white", lw=1, ls="--")


def part34(cam, ds, y, p, args, parts):
    rng = np.random.default_rng(args.seed)              # 시드 고정 난수
    masks = region_masks()                              # 구역 마스크
    means, faces, table = [], [], []                    # 그림용, 표용
    for b in range(NUM_CLASSES):                        # 나이대마다
        for ok in (True, False):                        # 맞음 / 틀림
            idx = np.where((y == b) & ((p == b) == ok))[0]   # 해당 사진
            idx = sample(idx, MAX_MEAN, rng)            # 최대 500장
            mc, mf, r = cam_stats(cam, ds, idx, p[idx], masks, args.device)   # Grad-CAM은 예측 나이대 기준
            if ok:                                      # 3번 평균 그림은 맞힌 사진만
                means.append(mc); faces.append(mf)
            table.append({"나이대": BAND_NAMES[b], "결과": "맞음" if ok else "틀림", "장수": len(idx),
                          **{f"{g}(%)": round(100 * v, 1) for g, v in zip(REGIONS, r.mean(axis=0))}})
            print(f"  {BAND_NAMES[b]} {'맞음' if ok else '틀림'} {len(idx)}장 완료")
    if "3" in parts:                                    # 3번 그림
        fig, axes = plt.subplots(2, NUM_CLASSES, figsize=(NUM_CLASSES * 2.2, 4.8))   # 위: 평균 얼굴+구역선, 아래: 평균 히트맵 겹침
        for b in range(NUM_CLASSES):
            face = Image.fromarray(np.uint8(faces[b]))  # 평균 얼굴
            axes[0][b].imshow(face); draw_regions(axes[0][b])   # 구역선 확인용
            axes[0][b].set_title(f"{BAND_NAMES[b]}\n평균 얼굴 + 구역", fontsize=9)
            heat = means[b] / (means[b].max() + 1e-8)   # 0~1로 (나이대 안에서 상대값)
            axes[1][b].imshow(overlay(face, heat)); draw_regions(axes[1][b])   # 평균 히트맵 겹침
            axes[1][b].set_title(f"평균 Grad-CAM ({int(table[b * 2]['장수'])}장)", fontsize=9)
            for ax in axes[:, b]:
                ax.axis("off")
        fig.suptitle("나이대별 맞힌 사진의 평균 Grad-CAM (최대 500장, 정답 나이대)", fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "mean_cam_by_band.png"), dpi=120); plt.close(fig)
    if "4" in parts:                                    # 4번 표·그림
        df = pd.DataFrame(table)                        # 나이대 × 맞음/틀림 표
        df.to_csv(os.path.join(args.out_dir, "region_ratio.csv"), index=False, encoding="utf-8-sig")   # 엑셀 한글용
        print(df.to_string(index=False))
        fig, ax = plt.subplots(figsize=(12, 4.5))       # 누적 막대
        labels = [f"{r['나이대']}\n{r['결과']}" for r in table]   # x축 이름
        bottom = np.zeros(len(table))                   # 누적 시작
        for g, color in zip(REGIONS, ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#9c9c9c"]):   # 구역별 색
            v = df[f"{g}(%)"].values                    # 이 구역 비율
            ax.bar(range(len(table)), v, bottom=bottom, color=color, label=g)
            bottom += v                                 # 위로 쌓기
        ax.set_xticks(range(len(table))); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("히트맵 에너지 비율 (%)"); ax.set_ylim(0, 100)
        ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "region_ratio.png"), dpi=120); plt.close(fig)

# -----------------------------------------------------------------------------
# ◀ [모듈 G-6 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   실행부
# =============================================================================
if __name__ == "__main__":
    args = parse_args()                                 # 명령줄 옵션
    assert "test" not in os.path.basename(args.csv).lower(), "시험 데이터는 쓰지 않습니다"   # 실수 방지
    plt.rcParams["font.family"] = "Malgun Gothic"       # 한글 글꼴 (윈도우)
    plt.rcParams["axes.unicode_minus"] = False          # 마이너스 기호 깨짐 방지
    torch.manual_seed(args.seed)                        # 시드
    os.makedirs(args.out_dir, exist_ok=True)            # 결과 폴더
    ds, y, p = load_data(args)                          # 데이터, 정답, 예측
    model = load_model(args)                            # 모델 (eval)
    cam = GradCAM(model, *target_layer(model, "small_cnn"))   # hook 걸기 (gap 입력 = conv4 블록 출력, 256×14×14)
    print(f"device {args.device} | 검증 {len(ds)}장 | 정확도 {(y == p).mean():.4f}")
    if "1" in args.parts:
        part1(model, cam, ds, y, p, args); print("1번 완료: correct_by_band.png")
    if "2" in args.parts:
        part2(model, cam, ds, y, p, args); print("2번 완료: confusion_pairs.png")
    if "3" in args.parts or "4" in args.parts:
        part34(cam, ds, y, p, args, args.parts); print("3·4번 완료")
    print("결과:", args.out_dir)

# #############################################################################
# 모듈 끝: gradcam_analysis.py
# #############################################################################
