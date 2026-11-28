# #############################################################################
#
#   AAF 다시 자르기 실험 — "얼굴 크기 = 나이 단서" 지름길을 직접 확인 (개발일지 18)
#   인공지능 4조 · 작은 CNN 담당: LIM
#
#   [하는 일] 검증 사진마다 세 가지 입력으로 예측 (--out-dir 폴더)
#     원본     : 학습 때와 같은 사진
#     다시자름 : 시연 app.detect_and_crop (Haar 상자 + v2 학습 사진 중앙값 여백) → 얼굴이 꽉 차게 → 224 확대
#     흐림대조 : 자르지 않고, 다시자름과 같은 픽셀 수로 줄였다가 224로 확대 → 해상도(흐림) 효과만 남김
#     → 다시자름 − 흐림대조 = 해상도를 맞춘 상태에서 "얼굴이 화면을 채우는 정도"만 바꾼 효과
#     FairFace·UTKFace에도 같은 처리를 해 대조군으로 씀 (원래 꽉 찬 얼굴이라 자르기 변화가 작음)
#
#   [결과] recrop_compare.csv (출처 × 나이대), recrop_per_image.csv, recrop_samples.png
#   [주의] 검증 CSV만 사용 (test.csv 사용 금지). 재학습 없음
#   [실행] 프로젝트 폴더에서
#     python <저장소>\작은CNN\aaf_recrop.py --weights results\small_cnn_v2_lr1e-3\small_cnn_best.pth
#            --csv splits_csv_v2\val.csv --image-root division_224AFAD-X
#            --pred-csv results\small_cnn_v2_lr1e-3\eval_val\predictions.csv
#            --out-dir results\small_cnn_v2_lr1e-3\recrop --device cpu
#
# #############################################################################

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "시연"))
from gradcam_analysis import load_data, load_model, BAND_NAMES   # 검증 데이터·모델 (predictions.csv 순서 점검 포함)
from CNN_4layers_unified import val_transform                    # Resize 224 → ToTensor → Normalize
from source_check import source_of, SOURCES                      # 출처 규칙 (개발일지 17)
import app                                                       # detect_and_crop (시연과 같은 자르기)

CONDS = ["원본", "다시자름", "흐림대조"]


def parse_args():
    p = argparse.ArgumentParser(description="AAF 다시 자르기 실험")
    p.add_argument("--weights", required=True)
    p.add_argument("--csv", required=True)              # val.csv (test.csv 금지)
    p.add_argument("--image-root", required=True)
    p.add_argument("--pred-csv", required=True)         # val predictions.csv (순서 점검용)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def three_inputs(path):                                 # → (원본, 다시자름, 흐림대조) PIL 또는 None, 자른 크기
    img = Image.open(path).convert("RGB")
    crop = app.detect_and_crop(img)                     # Haar 못 찾으면 None
    if crop is None:
        return None, None
    blur = img.resize(crop.size, Image.BILINEAR)        # 다시자름과 같은 픽셀 수로 줄임 (val_transform이 224로 다시 키움)
    return (img, crop, blur), crop.size


def predict_all(model, ds, device):                     # 사진마다 세 입력 예측
    rows, batch, meta = [], {c: [] for c in CONDS}, []

    def flush():
        with torch.no_grad():
            preds = {c: model(torch.stack(batch[c]).to(device)).argmax(1).cpu().numpy() for c in CONDS}
        for k, m in enumerate(meta):
            rows.append({**m, **{c: int(preds[c][k]) for c in CONDS}})
        for c in CONDS:
            batch[c].clear()
        meta.clear()

    for i, path in enumerate(ds.paths):
        ims, size = three_inputs(path)
        if ims is None:
            rows.append({"i": i, "path": path, "출처": source_of(path), "band": ds.labels[i], "검출": False})
            continue
        for c, im in zip(CONDS, ims):
            batch[c].append(val_transform(im))
        meta.append({"i": i, "path": path, "출처": source_of(path), "band": ds.labels[i], "검출": True,
                     "자른폭비": size[0] / 224, "자른높이비": size[1] / 224})
        if len(meta) == 128:
            flush()
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(ds)}")
    if meta:
        flush()
    return pd.DataFrame(rows).sort_values("i").reset_index(drop=True)   # 검출 실패 행이 먼저 들어가므로 원래 순서로


def summarize(d):                                       # 출처 × (나이대 + 전체) 표
    rows = []
    for s in [s for s in SOURCES if (d["출처"] == s).any()]:
        ds_ = d[d["출처"] == s]
        for name, g in [(BAND_NAMES[b], ds_[ds_["band"] == b]) for b in range(len(BAND_NAMES))] + [("전체", ds_)]:
            ok = g[g["검출"]]
            r = {"출처": s, "묶음": name, "장수": len(g), "검출": len(ok), "자른폭비_중앙값": ok["자른폭비"].median()}
            for c in CONDS:
                r[f"정답률_{c}"] = (ok[c] == ok["band"]).mean()            # 나이대 묶음 = 재현율, 전체 = 정확도
                r[f"예측차_{c}"] = (ok[c] - ok["band"]).mean()             # 예측 − 정답 (+면 나이 많게)
            r["바뀐비율_다시자름"] = (ok["다시자름"] != ok["원본"]).mean()
            r["바뀐비율_흐림대조"] = (ok["흐림대조"] != ok["원본"]).mean()
            r["자르기효과(다시자름−흐림대조)"] = (ok["다시자름"] - ok["흐림대조"]).mean()   # 해상도를 맞춘 순수 자르기 효과 (칸)
            rows.append(r)
    return pd.DataFrame(rows).round(3)


def sample_grid(d, out):                                # AAF 나이대별 1장씩: 원본 / 다시자름 / 흐림대조
    a = d[(d["출처"] == "AAF") & d["검출"]].groupby("band").head(1).sort_values("band")
    fig, ax = plt.subplots(3, len(a), figsize=(2 * len(a), 6.4))
    for j, (_, r) in enumerate(a.iterrows()):
        ims, _ = three_inputs(r["path"])
        for k, (c, im) in enumerate(zip(CONDS, ims)):
            ax[k][j].imshow(im.resize((224, 224), Image.BILINEAR)); ax[k][j].axis("off")
            ax[k][j].set_title(f"{c}: {BAND_NAMES[int(r[c])]}" if k else f"정답 {BAND_NAMES[int(r['band'])]}\n원본: {BAND_NAMES[int(r[c])]}", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
    assert "test" not in os.path.basename(args.csv), "test.csv 사용 금지"
    os.makedirs(args.out_dir, exist_ok=True)
    ds, y, p = load_data(args)
    model = load_model(args)
    d = predict_all(model, ds, args.device)
    d.to_csv(os.path.join(args.out_dir, "recrop_per_image.csv"), index=False, encoding="utf-8-sig")
    ok = d["검출"].values
    print(f"원본 예측 = predictions.csv 일치율: {(d.loc[ok, '원본'].values == p[ok]).mean():.4f}")
    assert (d.loc[ok, "원본"].values == p[ok]).mean() > 0.99, "원본 예측이 predictions.csv와 다름"
    s = summarize(d)
    s.to_csv(os.path.join(args.out_dir, "recrop_compare.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(s.to_string(index=False))
    sample_grid(d, os.path.join(args.out_dir, "recrop_samples.png"))
    print("결과:", args.out_dir)
