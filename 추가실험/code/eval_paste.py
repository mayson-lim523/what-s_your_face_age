"""여러 얼굴 사진에 얼마나 강한지 확인: 검증 사진 구석에 다른 사람 얼굴을 일부러 붙여 성능을 잰다.

평가 조건 4가지 (모든 모델에 똑같은 사진이 들어감 → 공정한 비교):
  원본            : 검증 사진 그대로
  원본+가리기     : 얼굴을 찾아 가장 큰 얼굴만 남기고 나머지 얼굴을 흐리게 가림 (방법 A를 원본에 적용해도 손해가 없는지 확인)
  얼굴 붙임       : 구석에 다른 사람 얼굴을 작게 붙임 (옆 사람이 걸친 상황)
  붙임+가리기     : 얼굴을 붙인 뒤 방법 A(가장 큰 얼굴만 남기고 가리기) 적용

사용 예:
python eval_paste.py --exp-dirs results/v2_convnext --split-csv splits_csv_v2/val.csv --image-root data_v2
python eval_paste.py --exp-dirs results/v2_convnext results/v2_convnext_paste --split-csv splits_csv_v2/val.csv --image-root data_v2
(빨리 해 보려면 --limit 3000, 예시 그림은 --examples 8)
결과: paste_eval.csv, paste_examples.png (현재 폴더)
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from tqdm import tqdm

from common import BANDS, build_model, build_transforms, compute_metrics, paste_face

CONDS = ["원본", "원본+가리기", "얼굴 붙임", "붙임+가리기"]


def mask_others(img, boxes, probs, min_prob=0.9):
    """가장 큰 얼굴만 남기고, 나머지 얼굴 상자는 강하게 흐려서 가린다. 가장 큰 얼굴과 많이 겹치는 상자는 건드리지 않는다."""
    if boxes is None:
        return img
    keep = [b for b, p in zip(boxes, probs) if p is not None and p >= min_prob]
    if len(keep) < 2:
        return img
    area = lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    keep.sort(key=area, reverse=True)
    main = keep[0]
    img = img.copy()
    W, H = img.size
    for b in keep[1:]:
        ix = max(0, min(b[2], main[2]) - max(b[0], main[0]))
        iy = max(0, min(b[3], main[3]) - max(b[1], main[1]))
        if ix * iy > 0.3 * area(b):  # 메인 얼굴과 많이 겹치면 메인을 가릴 수 있어 건너뜀
            continue
        w, h = b[2] - b[0], b[3] - b[1]
        box = (int(max(0, b[0] - 0.1 * w)), int(max(0, b[1] - 0.1 * h)), int(min(W, b[2] + 0.1 * w)), int(min(H, b[3] + 0.1 * h)))
        if box[2] > box[0] and box[3] > box[1]:
            img.paste(img.crop(box).filter(ImageFilter.GaussianBlur(12)), box[:2])
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dirs", nargs="+", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--seed", type=int, default=0, help="붙일 얼굴과 위치를 정하는 시드 (같으면 항상 같은 시험 사진)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--examples", type=int, default=8, help="예시 그림에 넣을 장수 (0이면 안 만듦)")
    ap.add_argument("--out", default="paste_eval.csv")
    a = ap.parse_args()

    from facenet_pytorch import MTCNN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=True, device=device, post_process=False, min_face_size=20)
    tf = build_transforms(train=False)

    df = pd.read_csv(a.split_csv).reset_index(drop=True)
    if a.limit:
        df = df.sample(n=min(a.limit, len(df)), random_state=0).reset_index(drop=True)
    paths = [os.path.join(a.image_root, p) for p in df["path"]]
    y = df["band"].astype(int).to_numpy()
    n = len(df)

    models = []
    for exp in a.exp_dirs:
        ck = torch.load(os.path.join(exp, "best.pt"), map_location=device, weights_only=False)
        m = build_model(ck["model_name"], pretrained=False).to(device)
        m.load_state_dict(ck["model"]); m.eval()
        models.append((os.path.basename(os.path.normpath(exp)), m))
    preds = {(name, c): [] for name, _ in models for c in CONDS}
    examples = []

    for s in tqdm(range(0, n, a.batch), desc="평가"):
        idx = range(s, min(s + a.batch, n))
        orig, pasted = [], []
        for i in idx:
            img = Image.open(paths[i]).convert("RGB").resize((224, 224))
            rng = random.Random(a.seed * 1_000_003 + i)
            j = rng.randrange(n - 1); j = j + 1 if j >= i else j  # 자기 자신이 아닌 다른 사진
            other = Image.open(paths[j]).convert("RGB")
            orig.append(img); pasted.append(paste_face(img, other, rng))
        imgs = {"원본": orig, "얼굴 붙임": pasted}
        for src, dst in [("원본", "원본+가리기"), ("얼굴 붙임", "붙임+가리기")]:
            boxes, probs = mtcnn.detect(imgs[src])
            imgs[dst] = [mask_others(im, b, p) for im, b, p in zip(imgs[src], boxes, probs)]
        if len(examples) < a.examples:
            for k in range(min(a.examples - len(examples), len(orig))):
                examples.append((orig[k], pasted[k], imgs["붙임+가리기"][k], int(y[s + k])))
        with torch.no_grad(), torch.autocast(device_type=device, enabled=(device == "cuda")):
            for c in CONDS:
                x = torch.stack([tf(im) for im in imgs[c]]).to(device)
                for name, m in models:
                    preds[(name, c)].append(m(x).argmax(1).cpu().numpy())

    rows = []
    for name, _ in models:
        for c in CONDS:
            mt = compute_metrics(y, np.concatenate(preds[(name, c)]))
            rows.append({"실험": name, "조건": c, "정확도": mt["acc"], "한칸차이정확도": mt["off1_acc"], "macroF1": mt["macro_f1"]})
    t = pd.DataFrame(rows)
    t.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(t.round(3).to_string(index=False))
    print("\n[macro-F1 표]")
    print(t.pivot(index="조건", columns="실험", values="macroF1").reindex(CONDS).round(3).to_string())
    print(f"\n저장: {a.out}")

    if examples:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]:
            if any(f == x.name for x in font_manager.fontManager.ttflist):
                plt.rcParams["font.family"] = f; break
        fig, axes = plt.subplots(len(examples), 3, figsize=(7.5, 2.5 * len(examples)))
        axes = np.atleast_2d(axes)
        for r, (o, p, q, lab) in enumerate(examples):
            for c, (im, title) in enumerate([(o, f"원본 (정답 {BANDS[lab]})"), (p, "얼굴 붙임"), (q, "붙임+가리기")]):
                axes[r, c].imshow(im); axes[r, c].set_title(title, fontsize=9); axes[r, c].axis("off")
        fig.tight_layout(); fig.savefig("paste_examples.png", dpi=110)
        print("예시 그림: paste_examples.png")


if __name__ == "__main__":
    main()
