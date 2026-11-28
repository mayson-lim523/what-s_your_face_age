"""흐린 사진에 얼마나 강한지 확인: 검증 사진을 일부러 흐리게 만들어 흐림 정도별 성능을 잰다.

모든 모델에 '똑같이 흐리게 만든' 검증 사진을 넣기 때문에, 흐림 증강 전후 모델을 공정하게 비교할 수 있다.
흐림 정도: 원본 / 가우시안 블러 σ=1, 2, 3 / 저해상도(112, 56픽셀로 줄였다가 다시 224로)

사용 예 (여러 모델 한 번에):
python eval_blur.py --exp-dirs results/v2_convnext results/v2_convnext_blur --split-csv splits_csv_v2/val.csv --image-root data_v2
결과: blur_eval.csv (현재 폴더, --out으로 변경 가능)
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from common import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, FaceDataset, build_model, compute_metrics

LEVELS = ["원본", "블러 σ=1", "블러 σ=2", "블러 σ=3", "저해상도 112px", "저해상도 56px"]


def make_transform(level):
    from torchvision import transforms as T
    steps = [T.Resize((IMG_SIZE, IMG_SIZE))]
    if level.startswith("블러"):
        sigma = float(level.split("=")[1])
        k = int(2 * round(3 * sigma) + 1)  # 흐림 정도에 맞는 창 크기
        steps.append(T.GaussianBlur(kernel_size=k, sigma=sigma))
    elif level.startswith("저해상도"):
        px = int(level.split()[1].replace("px", ""))
        steps += [T.Resize(px), T.Resize((IMG_SIZE, IMG_SIZE))]
    steps += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dirs", nargs="+", required=True, help="best.pt가 있는 실험 폴더들")
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="blur_eval.csv")
    a = ap.parse_args()

    from torch.utils.data import DataLoader
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(a.split_csv)
    y = df["band"].astype(int).to_numpy()

    rows = []
    for exp in a.exp_dirs:
        ck = torch.load(os.path.join(exp, "best.pt"), map_location=device, weights_only=False)
        model = build_model(ck["model_name"], pretrained=False).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        name = os.path.basename(os.path.normpath(exp))
        for level in LEVELS:
            dl = DataLoader(FaceDataset(df, a.image_root, make_transform(level)), 128, num_workers=a.workers)
            preds = []
            with torch.no_grad(), torch.autocast(device_type=device, enabled=(device == "cuda")):
                for x, _ in dl:
                    preds.append(model(x.to(device)).argmax(1).cpu().numpy())
            m = compute_metrics(y, np.concatenate(preds))
            rows.append({"실험": name, "흐림 정도": level, "정확도": m["acc"], "한칸차이정확도": m["off1_acc"], "macroF1": m["macro_f1"]})
            print(f"{name:>22} | {level:<12} | 정확도 {m['acc']:.3f} | 한 칸 {m['off1_acc']:.3f} | F1 {m['macro_f1']:.3f}")

    t = pd.DataFrame(rows)
    t.to_csv(a.out, index=False, encoding="utf-8-sig")
    print("\n[macro-F1 표]")
    print(t.pivot(index="흐림 정도", columns="실험", values="macroF1").reindex(LEVELS).round(3).to_string())
    print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
