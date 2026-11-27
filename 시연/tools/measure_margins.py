# #############################################################################
#
#   얼굴 자르는 여백 측정 — 학습 사진에서 Haar 얼굴 상자의 크기·위치 재기
#   인공지능 4조 · 시연 화면 담당: LIM
#
#   [하는 일] 학습 사진을 나이대별로 --per-band장 뽑아 app.detect_face(시연과 같은 검출)를 돌리고,
#            상자 크기·중심·상자 밖 여백(왼·위·오른·아래, 상자 크기 대비)의 평균·사분위·중앙값을 출력
#            → 중앙값이 app.py의 MARGINS 근거 (README 5-1 표)
#   [실행] 사진·CSV가 있는 프로젝트 폴더에서
#     python <저장소>\시연\tools\measure_margins.py --csv splits_csv_v2\train.csv --image-root division_224AFAD-X
#   [README 값] 기본 옵션(나이대별 300장, seed 0) → 검출 1,795/2,100, 상자 폭 0.964, 중심 y 0.420,
#              여백 왼 0.014 / 위 −0.065 / 오른 0.028 / 아래 0.105
#
# #############################################################################

import argparse                                         # 명령줄 옵션
import os                                               # 경로
import sys                                              # import 경로

import pandas as pd                                     # 표
from PIL import Image                                   # 사진 열기

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # 시연/ 폴더 → app.py import
import app                                              # detect_face (시연 화면과 같은 검출)


def parse_args():                                       # 명령줄 옵션
    p = argparse.ArgumentParser(description="학습 사진의 Haar 얼굴 상자 크기·위치 측정")
    p.add_argument("--csv", required=True)              # 학습 CSV (열: path, band)
    p.add_argument("--image-root", required=True)       # CSV path 기준 폴더
    p.add_argument("--per-band", type=int, default=300) # 나이대별 장수
    p.add_argument("--seed", type=int, default=0)       # 뽑기 시드
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert "test" not in os.path.basename(args.csv).lower(), "시험 데이터는 쓰지 않습니다"   # 실수 방지
    df = pd.read_csv(args.csv).groupby("band").sample(args.per_band, random_state=args.seed)   # 나이대별 같은 장수
    rows = []                                           # 검출된 사진별 측정값
    for path, band in zip(df["path"], df["band"]):
        img = Image.open(os.path.join(args.image_root, path))   # 원본 (v2는 224×224)
        box = app.detect_face(img)                      # (x, y, w, h) 원본 좌표 또는 None
        if box is None:                                 # 못 찾으면 제외
            continue
        x, y, w, h = box
        W, H = img.size
        rows.append({"band": band,
                     "상자폭비": w / W, "상자높이비": h / H,                     # 상자 크기 ÷ 사진 크기
                     "중심x": (x + w / 2) / W, "중심y": (y + h / 2) / H,       # 상자 중심 (0~1)
                     "왼": x / w, "위": y / h,                                 # 상자 밖 여백 ÷ 상자 크기 (음수 = 사진이 상자보다 좁음)
                     "오": (W - x - w) / w, "아래": (H - y - h) / h})
    d = pd.DataFrame(rows)
    print(f"검출 {len(d)}/{len(df)} ({len(d) / len(df):.0%})")
    cols = ["상자폭비", "상자높이비", "중심x", "중심y", "왼", "위", "오", "아래"]
    print(d[cols].describe(percentiles=[.25, .5, .75]).loc[["mean", "25%", "50%", "75%"]].round(3).to_string())
    print("\n나이대별 중앙값")
    print(d.groupby("band")[["상자폭비", "중심y", "왼", "위", "오", "아래"]].median().round(3).to_string())
    m = d[["왼", "위", "오", "아래"]].median()          # 중앙값 여백
    # 측정값 = 사진 가장자리가 상자에서 떨어진 거리 ÷ 상자 크기 = detect_and_crop 여백과 같은 뜻 (음수 = 상자 안쪽)
    print(f"\n제안 MARGINS (왼·오른은 평균): ({(m['왼'] + m['오']) / 2:.3f}, {m['위']:.3f}, {(m['왼'] + m['오']) / 2:.3f}, {m['아래']:.3f})"
          f"  ← app.py 현재 값 {app.MARGINS}")

# #############################################################################
# 모듈 끝: measure_margins.py
# #############################################################################
