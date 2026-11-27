# #############################################################################
#
#   시연 검출·자르기 점검 — 원본 그대로 vs 검출·자르기를 거친 예측이 얼마나 같은지
#   인공지능 4조 · 시연 화면 담당: LIM
#
#   [하는 일] 검증 사진마다 원본(224) 예측 1위와, 아래 두 변형을 app.detect_and_crop → app.predict 한 1위를 비교
#     ① 2배 확대 (448×448)
#     ② 448 회색 바탕 가운데에 원본 (얼굴 주변에 배경이 있는 사진 흉내)
#     각각 새 여백(app.MARGINS)과 예전 여백(사방 0.15)으로 → README 5-1 점검 표
#   [실행] 사진·CSV가 있는 프로젝트 폴더에서 (CPU로 1~2분)
#     python <저장소>\시연\tools\compare_crop.py --csv splits_csv_v2\val.csv --image-root division_224AFAD-X
#   [README 값] 기본 옵션 → 10장(seed 0): 확대 7·8, 바탕 5·2 / 210장(나이대별 30, seed 1): 확대 145·153, 바탕 132·106
#   [주의] 검증 데이터만 사용 (test.csv 금지)
#
# #############################################################################

import argparse                                         # 명령줄 옵션
import os                                               # 경로
import sys                                              # import 경로

import pandas as pd                                     # 표
from PIL import Image                                   # 사진

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # 시연/ → app.py import
import app                                              # load_model, detect_and_crop, predict, MARGINS (시연과 같은 함수)

OLD_MARGINS = (0.15, 0.15, 0.15, 0.15)                  # 예전 여백 (사방 0.15)


def parse_args():                                       # 명령줄 옵션
    p = argparse.ArgumentParser(description="원본 vs 검출·자르기 예측 비교")
    p.add_argument("--csv", required=True)              # 검증 CSV
    p.add_argument("--image-root", required=True)       # 사진 기준 폴더
    p.add_argument("--weights", default=app.DEFAULT_WEIGHTS["small_cnn"])   # 작은 CNN 가중치 (기본 v2)
    p.add_argument("--device", default="cpu")           # 장치 (몇백 장이라 CPU로 충분)
    return p.parse_args()


def top1(model, img, device):                           # 예측 1위 나이대 이름
    p = app.predict(model, img, device)
    return max(p, key=p.get)


def enlarge(img):                                       # ① 2배 확대
    return img.resize((448, 448), Image.BICUBIC)


def on_canvas(img):                                     # ② 448 회색 바탕 가운데에 원본
    c = Image.new("RGB", (448, 448), (128, 128, 128))
    c.paste(img, (112, 112))
    return c


def compare(model, df, args, show=False):               # df 사진들로 표 한 묶음 출력
    rows = []                                           # 사진별 1위
    for path, band in zip(df["path"], df["band"]):
        img = Image.open(os.path.join(args.image_root, path)).convert("RGB")
        r = {"파일": os.path.basename(path), "정답": app.BAND_NAMES[band], "원본": top1(model, img, args.device)}
        for tname, tf in [("확대", enlarge), ("바탕", on_canvas)]:
            for mname, m in [("새", app.MARGINS), ("예전", OLD_MARGINS)]:
                face = app.detect_and_crop(tf(img), m)  # 시연과 같은 검출·자르기
                r[f"{tname}_{mname}"] = top1(model, face, args.device) if face is not None else "검출실패"
        rows.append(r)
    d = pd.DataFrame(rows)
    if show:                                            # 10장은 사진별로 보여 줌
        print(d.to_string(index=False))
    for tname in ["확대", "바탕"]:
        det = (d[f"{tname}_새"] != "검출실패").sum()    # 검출 장수 (두 여백 모두 같은 detect_face라 같음)
        same = {m: (d[f"{tname}_{m}"] == d["원본"]).sum() for m in ["새", "예전"]}
        print(f"  {tname}: 검출 {det}/{len(d)} | 원본과 1위 같음 — 새 여백 {same['새']} ({same['새'] / det:.0%}), "
              f"예전 0.15 {same['예전']} ({same['예전'] / det:.0%})  (괄호 = 검출된 것 중)")
    print(f"  원본 정답 {(d['원본'] == d['정답']).sum()}/{len(d)}")


if __name__ == "__main__":
    args = parse_args()
    assert "test" not in os.path.basename(args.csv).lower(), "시험 데이터는 쓰지 않습니다"
    model = app.load_model("small_cnn", args.weights, args.device)   # eval 상태 모델
    val = pd.read_csv(args.csv)
    print(f"가중치 {args.weights} | 새 여백 {app.MARGINS} | 예전 {OLD_MARGINS}")
    print("\n== 10장 (나이대별 2장 → 10장, seed 0) ==")
    compare(model, val.groupby("band").sample(2, random_state=0).sample(10, random_state=0), args, show=True)
    print("\n== 210장 (나이대별 30장, seed 1) ==")
    compare(model, val.groupby("band").sample(30, random_state=1), args)

# #############################################################################
# 모듈 끝: compare_crop.py
# #############################################################################
