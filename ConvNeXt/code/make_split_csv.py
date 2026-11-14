"""팀원이 사진을 training / validation / testing 폴더로 나눠 준 경우,
step4_train.py가 읽을 수 있는 CSV(train.csv, val.csv, test.csv)를 만든다.

지원하는 폴더 모양:
  <root>/training/<나이대 폴더>/사진.jpg   (나이대 폴더 이름: 0~6, 또는 0-9세/10대/.../60세 이상, 또는 0-9/10-19/.../60+ 등)

사용 예:
python make_split_csv.py --root split --out-dir splits_csv
그다음 학습:
python step4_train.py --name base_resnet50 --model resnet50 --train-csv splits_csv/train.csv --val-csv splits_csv/val.csv --image-root split
"""
import argparse
import os
import re

import pandas as pd

from common import BANDS

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLIT_NAMES = {"train": ["training", "train"], "val": ["validation", "val", "valid"], "test": ["testing", "test"]}


def folder_to_band(name: str):
    n = name.strip()
    if n.isdigit() and 0 <= int(n) <= 6:          # 0~6 번호
        return int(n)
    for i, b in enumerate(BANDS):                   # 0-9세, 10대 ...
        if n.replace(" ", "") == b.replace(" ", ""):
            return i
    m = re.match(r"^(\d+)", n)                      # 0 9, 0-9, 10 19, 20s, 60over, 60+ ... (앞 숫자만 사용)
    if m:
        return min(int(m.group(1)) // 10, 6)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="training/validation/testing 폴더가 들어 있는 폴더")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    for key, names in SPLIT_NAMES.items():
        sub = next((n for n in names if os.path.isdir(os.path.join(a.root, n))), None)
        if sub is None:
            print(f"[건너뜀] {names} 폴더가 없음")
            continue
        rows, unknown = [], set()
        for cls in sorted(os.listdir(os.path.join(a.root, sub))):
            cdir = os.path.join(a.root, sub, cls)
            if not os.path.isdir(cdir):
                continue
            band = folder_to_band(cls)
            if band is None:
                unknown.add(cls)
                continue
            for dirpath, _, files in os.walk(cdir):
                for f in files:
                    if f.lower().endswith(IMG_EXT):
                        rel = os.path.relpath(os.path.join(dirpath, f), a.root).replace("\\", "/")
                        rows.append({"path": rel, "band": band, "source": "unknown", "race_group": "unknown"})
        if unknown:
            raise SystemExit(f"나이대를 알 수 없는 폴더 이름: {sorted(unknown)} → 이름을 알려주세요")
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(a.out_dir, f"{key}.csv"), index=False)
        counts = df["band"].value_counts().sort_index()
        print(f"\n{sub} → {key}.csv ({len(df):,}장)")
        for b, c in counts.items():
            print(f"  {BANDS[b]}: {c:,}")


if __name__ == "__main__":
    main()
