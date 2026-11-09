"""6단계: Grad-CAM 열지도 만들기. 나이대마다 사진 몇 장을 골라 '모델이 어디를 봤는지' 그림으로 저장한다.

사용 예:
python step6_gradcam.py --exp-dir results/base_resnet50 --split-csv splits/val_small.csv --image-root processed --per-band 4

결과: results/<실험>/gradcam/gradcam_grid.png (행 = 실제 나이대, 칸 아래 = 예측)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from common import BANDS, IMG_SIZE, build_model, build_transforms, gradcam_target_layer
from step5_evaluate import set_korean_font


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--per-band", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(a.exp_dir, "best.pt"), map_location=device, weights_only=False)
    name = ck["model_name"]
    model = build_model(name, pretrained=False).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    cam = GradCAM(model=model, target_layers=[gradcam_target_layer(model, name)])
    tf = build_transforms(train=False)

    df = pd.read_csv(a.split_csv)
    picks = df.sample(frac=1, random_state=a.seed).groupby("band").head(a.per_band)

    set_korean_font()
    bands = sorted(picks["band"].unique())
    fig, axes = plt.subplots(len(bands), a.per_band, figsize=(2.6 * a.per_band, 2.9 * len(bands)), squeeze=False)
    for r, band in enumerate(bands):
        rows = picks[picks["band"] == band]
        for c in range(a.per_band):
            ax = axes[r, c]; ax.axis("off")
            if c >= len(rows):
                continue
            img = Image.open(os.path.join(a.image_root, rows.iloc[c]["path"])).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            x = tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = int(model(x).argmax(1))
            heat = cam(input_tensor=x)[0]  # 예측한 나이대를 기준으로 열지도 계산
            overlay = show_cam_on_image(np.asarray(img, dtype=np.float32) / 255.0, heat, use_rgb=True)
            ax.imshow(overlay)
            ax.set_title(f"예측 {BANDS[pred]}", fontsize=9, color="black" if pred == band else "red")
        axes[r, 0].text(-0.08, 0.5, f"실제 {BANDS[band]}", transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=10)

    out = os.path.join(a.exp_dir, "gradcam")
    os.makedirs(out, exist_ok=True)
    fig.suptitle(f"Grad-CAM ({name}) · 빨간 제목 = 틀린 예측")
    fig.tight_layout(); fig.savefig(os.path.join(out, "gradcam_grid.png"), dpi=150); plt.close(fig)
    print("저장:", os.path.join(out, "gradcam_grid.png"))


if __name__ == "__main__":
    main()
