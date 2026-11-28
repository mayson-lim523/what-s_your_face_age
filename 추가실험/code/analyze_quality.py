"""흐린 사진·얼굴이 여러 개인 사진에서 정말 정확도가 떨어지는지 숫자로 확인한다.

평가(step5)로 만든 predictions.csv의 사진마다
  - 선명도(라플라시안 분산, 작을수록 흐림)
  - 얼굴 개수(MTCNN, 확신도 0.9 이상)와 두 번째 얼굴 크기
를 재고, 그룹별 정확도를 비교한다.

사용 예:
python analyze_quality.py --pred results/m_convnext_in1k/eval_val/predictions.csv --image-root split_result_224
(빨리 해 보려면 --limit 3000)

결과: 같은 폴더에 quality_per_image.csv (사진별 값), quality_summary.csv (그룹별 정확도)
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


def sharpness(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="step5가 만든 predictions.csv")
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--limit", type=int, default=0, help="앞에서부터 몇 장만 (0이면 전체)")
    ap.add_argument("--min-face-prob", type=float, default=0.90)
    a = ap.parse_args()

    from facenet_pytorch import MTCNN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=True, device=device, post_process=False, min_face_size=15)

    df = pd.read_csv(a.pred)
    if a.limit:
        df = df.sample(n=min(a.limit, len(df)), random_state=0).reset_index(drop=True)

    sharp, n_faces, second_ratio = [], [], []
    for p in tqdm(df["path"], desc="사진 분석"):
        img = Image.open(os.path.join(a.image_root, p)).convert("RGB")
        arr = np.asarray(img)
        sharp.append(sharpness(arr))
        boxes, probs = mtcnn.detect(img)
        if boxes is None:
            n_faces.append(0); second_ratio.append(0.0); continue
        areas = sorted([(b[2] - b[0]) * (b[3] - b[1]) for b, pr in zip(boxes, probs)
                        if pr is not None and pr >= a.min_face_prob], reverse=True)
        n_faces.append(len(areas))
        second_ratio.append(float(areas[1] / areas[0]) if len(areas) >= 2 else 0.0)

    df["선명도"] = sharp
    df["얼굴수"] = n_faces
    df["둘째얼굴크기비"] = second_ratio  # 두 번째 얼굴 넓이 / 가장 큰 얼굴 넓이

    # 선명도는 4등분 (1 = 가장 흐림)
    df["선명도구간"] = pd.qcut(df["선명도"], 4, labels=["1 가장 흐림", "2", "3", "4 가장 선명"])
    df["얼굴구간"] = pd.cut(df["얼굴수"], [-1, 0, 1, 100], labels=["얼굴 못 찾음", "1명", "2명 이상"])

    rows = []
    for col in ["선명도구간", "얼굴구간"]:
        g = df.groupby(col, observed=True).agg(장수=("correct", "size"), 정확도=("correct", "mean"),
                                               한칸차이정확도=("off1", "mean"))
        g.index = [f"{col}: {i}" for i in g.index]
        rows.append(g)
    allrow = pd.DataFrame({"장수": [len(df)], "정확도": [df["correct"].mean()],
                           "한칸차이정확도": [df["off1"].mean()]}, index=["전체"])
    summary = pd.concat([allrow] + rows).round(3)

    out = os.path.dirname(a.pred)
    df.to_csv(os.path.join(out, "quality_per_image.csv"), index=False, encoding="utf-8-sig")
    summary.to_csv(os.path.join(out, "quality_summary.csv"), encoding="utf-8-sig")
    print(summary.to_string())
    print(f"\n저장: {out}\\quality_summary.csv, quality_per_image.csv")


if __name__ == "__main__":
    main()
