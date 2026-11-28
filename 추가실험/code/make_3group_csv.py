"""7개 나이대 CSV를 3그룹(소아청소년 / 성인 / 노인) CSV로 바꾼다. 사진과 분할은 그대로, 정답 번호만 바꾼다.

  0-9세, 10대        → 0 소아청소년 (근사: 원래 기준은 만 19세 미만)
  20대 ~ 50대        → 1 성인
  60세 이상          → 2 노인        (근사: 원래 기준은 만 65세 이상)

사용 예:
python make_3group_csv.py --in-dir splits_csv_v2 --out-dir splits_csv_3g
"""
import argparse
import os

import pandas as pd

GROUP_OF_BAND = [0, 0, 1, 1, 1, 1, 2]
GROUPS = ["소아청소년(0~10대)", "성인(20~50대)", "노인(60세 이상)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="7개 나이대 CSV 폴더 (train/val/test.csv)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for name in ["train", "val", "test"]:
        df = pd.read_csv(os.path.join(a.in_dir, f"{name}.csv"))
        if df["band"].max() > 6:
            raise SystemExit(f"{name}.csv의 band 값이 0~6 범위가 아닙니다.")
        df["band7"] = df["band"]  # 원래 나이대 번호도 남겨 둠
        df["band"] = df["band"].map(lambda b: GROUP_OF_BAND[int(b)])
        df.to_csv(os.path.join(a.out_dir, f"{name}.csv"), index=False, encoding="utf-8-sig")
        cnt = df["band"].value_counts().sort_index()
        print(f"{name}.csv ({len(df):,}장): " + ", ".join(f"{GROUPS[i]} {cnt.get(i, 0):,}" for i in range(3)))
    print(f"저장: {a.out_dir}")


if __name__ == "__main__":
    main()
