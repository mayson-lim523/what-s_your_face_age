"""7개 나이대 예측을 3개 그룹(소아청소년 / 성인 / 노인)으로 합쳐서 다시 채점한다. (재학습 없음)

나이대 → 그룹 (데이터에 정확한 나이가 없어 10살 단위 경계로 근사):
  0-9세, 10대        → 소아청소년 (근사: 원래 기준은 만 19세 미만)
  20대 ~ 50대        → 성인
  60세 이상          → 노인        (근사: 원래 기준은 만 65세 이상)

사용 예 (여러 모델 한 번에):
python eval_3groups.py --pred results/base_resnet50/eval_val/predictions.csv results/m_convnext_in1k/eval_val/predictions.csv --names ResNet50 ConvNeXt
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

GROUP_OF_BAND = [0, 0, 1, 1, 1, 1, 2]
GROUPS = ["소아청소년", "성인", "노인"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    a = ap.parse_args()

    rows = []
    for path, name in zip(a.pred, a.names):
        d = pd.read_csv(path)
        y = d["band"].map(lambda b: GROUP_OF_BAND[int(b)]).to_numpy()
        p = d["pred"].map(lambda b: GROUP_OF_BAND[int(b)]).to_numpy()
        cm = confusion_matrix(y, p, labels=[0, 1, 2])
        recall = cm.diagonal() / cm.sum(1)
        rows.append({"모델": name, "7클래스 정확도": (d["pred"] == d["band"]).mean(),
                     "3그룹 정확도": (y == p).mean(), "3그룹 macro-F1": f1_score(y, p, average="macro"),
                     **{f"{g} 맞힌 비율": r for g, r in zip(GROUPS, recall)}})
        print(f"\n[{name}] 3그룹 혼동 행렬 (행 = 실제, 열 = 예측)")
        print(pd.DataFrame(cm, index=GROUPS, columns=GROUPS).to_string())

    t = pd.DataFrame(rows).round(3)
    print("\n" + t.to_string(index=False))
    t.to_csv("eval_3groups.csv", index=False, encoding="utf-8-sig")
    print("\n저장: eval_3groups.csv")


if __name__ == "__main__":
    main()
