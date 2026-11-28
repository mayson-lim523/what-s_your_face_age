# #############################################################################
#
#   얼굴 나이대 분류 시연 — 작은 CNN · ConvNeXt-Tiny · ResNet-50 + Grad-CAM
#   인공지능 4조 · 담당: LIM
#
#   [실행] 저장소 최상위에서:  .venv\Scripts\python 작은CNN\app_gradio.py   → http://127.0.0.1:7860
#   [점검] .venv\Scripts\python 작은CNN\app_gradio.py --selftest --image-root <division_224AFAD-X 폴더>
#   [팀원 공유] 기본은 이 PC에서만(127.0.0.1). 필요할 때만 켬
#     --lan              같은 와이파이·네트워크에서 접속 허용(0.0.0.0), 실행할 때 이 PC의 IP 주소 출력
#     --share            Gradio 공유 링크(https://xxxx.gradio.live) — 인터넷 어디서나 접속 가능
#     --auth 아이디:비번  로그인 걸기. --share·--lan인데 --auth가 없으면 경고
#
#   [흐름] 사진 → face_align.crop_face (YuNet 검출 → 가장 큰 얼굴 → (정렬) → 학습 사진과 같은 여백으로 자르기 → 224)
#          → gradcam_utils.preprocess (팀 검증 전처리) → 모델 예측 + Grad-CAM (예측 1위 나이대 기준)
#          → face_regions: YuNet 5점으로 얼굴 영역 나누기 → 영역별 주목 비중 (+ 가림 실험 체크 시 영역별 블러 후 재예측)
#            5점은 설명용으로만 쓰고 모델 입력에는 넣지 않음
#
#   [사진을 디스크에 남기지 않기]
#     - Gradio는 업로드·웹캠 사진과 출력 그림을 임시 폴더에 파일로 둔다 (브라우저에 보내려면 필요)
#     - 그 폴더를 이 앱 전용 새 임시 폴더로 바꾸고(GRADIO_TEMP_DIR, gradio import 전에),
#       delete_cache로 1분마다 1분 지난 파일 삭제 + 앱 종료 때 폴더째 삭제
#     - 앱 코드 자체는 사진을 메모리(PIL)로만 다룬다. 기본은 127.0.0.1에서만 열림 (--share/--lan일 때만 밖에서 접속)
#     - --share를 켜면 사진이 Gradio 공유 서버(gradio.live)를 거쳐 전달된다. 파일로 남는 곳은 여전히 이 PC 임시 폴더뿐
#     - Gradio 사용 통계 전송도 끔
#
# #############################################################################

import argparse
import atexit
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# 주의: 앱을 2개 동시에 띄우면 나중 것이 먼저 것의 폴더를 지움 — 시연은 1개만 띄우므로 이대로 둠
for old in Path(tempfile.gettempdir()).glob("face_age_gradio_*"):   # 강제 종료로 남은 예전 실행의 임시 폴더 삭제
    shutil.rmtree(old, ignore_errors=True)
TMP = Path(tempfile.mkdtemp(prefix="face_age_gradio_"))
os.environ["GRADIO_TEMP_DIR"] = str(TMP)
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
atexit.register(shutil.rmtree, TMP, ignore_errors=True)

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import face_align as fa  # noqa: E402
import face_regions as fr  # noqa: E402
from gradcam_utils import BANDS, GradCAM, MODEL_NAMES, load_model, overlay, preprocess  # noqa: E402

TITLES = {"small_cnn": "작은 CNN", "convnext_tiny": "ConvNeXt-Tiny", "resnet50": "ResNet-50"}
NOTICE = "예측은 겉보기 나이 기준이며 참고용입니다. 업로드한 사진은 저장하지 않습니다."
REGION_NOTICE = "영역 구분은 설명용이며 예측에는 쓰이지 않습니다."


# =============================================================================
#   ▶ [모듈 1] 모델 3개를 앱 시작 때 한 번만 읽기 (GPU 있으면 GPU)
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# GPU 기본값인 TF32(합성곱을 19비트 정밀도로 계산)를 끔. 켜 두면 사진 1장/5장 묶음에 따라 ResNet-50 확률이
# 최대 0.0018 달라짐(점검에서 확인). 시연은 1장씩이라 속도 차이가 거의 없으므로 정확한 32비트로 계산
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
CAMS = {n: GradCAM(load_model(n, device=DEVICE), n) for n in MODEL_NAMES}
for _cam in CAMS.values():                      # 미리 한 번 실행 — 안 하면 첫 사진만 3~4초 걸림 (CUDA 첫 실행 준비)
    _cam(torch.zeros(1, 3, 224, 224))
    with torch.no_grad():                       # 가림 실험 배치(영역 9장)도 첫 실행이 느려서 같이 준비
        _cam.model(torch.zeros(len(fr.REGIONS), 3, 224, 224, device=DEVICE))
fa.crop_face(Image.new("RGB", (640, 480)))      # YuNet 검출기도 첫 실행 준비 (빈 사진이라 결과는 없음)


# =============================================================================
#   ▶ [모듈 2] 예측 — 잘라낸 224 얼굴 1장 → (나이대별 확률 dict, Grad-CAM 겹친 그림)
#   - selftest에서 gradcam_utils로 직접 계산한 값과 비교하는 함수가 바로 이것
# =============================================================================

def predict(face, name):
    heat, _, prob = CAMS[name](preprocess([face]))
    return {BANDS[i]: float(prob[0, i]) for i in range(len(BANDS))}, overlay(face, heat[0]), heat[0]


# =============================================================================
#   ▶ [모듈 3] 화면 처리 — 사진 → 얼굴 자르기 → 모델 1개 또는 3개
#   - 출력 순서: 안내 문구, 잘라낸 얼굴, (확률, Grad-CAM, 제목, 글 설명) × 3칸. 단일 모드면 1칸만 보임
#   - 글 설명: 지금 보이는 Grad-CAM(예측 1위 기준)의 영역별 주목 비중 + (켜면) 가림 실험 표
# =============================================================================

def explain(face, labels, name, label, heat, occlude):
    band = max(label, key=label.get)
    text = fr.share_text(fr.attention_share(heat, labels))
    if occlude:
        p = fr.occlusion(CAMS[name].model, face, labels, BANDS.index(band), preprocess)
        text += "\n\n" + fr.occlusion_text(label[band], p, band)
    return text


def run(image, model_title, compare, align, occlude=False):
    empty = [None, None, gr.update(visible=False), None] * 3
    if image is None:
        return ["사진을 올리거나 웹캠으로 찍어 주세요.", None] + empty
    face, n, lm = fa.crop_face(image, align=align, return_landmarks=True)
    if face is None:
        return ["얼굴을 찾지 못했습니다. 얼굴이 정면으로 잘 보이는 밝은 사진으로 다시 시도해 주세요.", None] + empty
    names = MODEL_NAMES if compare else [next(k for k, v in TITLES.items() if v == model_title)]
    t0 = time.perf_counter()
    labels = fr.face_regions(lm)
    out = []
    for name in MODEL_NAMES:
        if name in names:
            label, cam, heat = predict(face, name)
            out += [label, cam, gr.update(visible=True, value=f"### {TITLES[name]}"),
                    explain(face, labels, name, label, heat, occlude)]
        else:
            out += [None, None, gr.update(visible=False), None]
    msg = f"얼굴 {n}개 중 가장 큰 얼굴" if n > 1 else "얼굴 1개 검출"
    return [f"{msg} · 처리 {time.perf_counter() - t0:.2f}초 ({DEVICE})", face] + out


def build():
    with gr.Blocks(title="얼굴 나이대 분류", delete_cache=(60, 60)) as demo:
        gr.Markdown("# 얼굴 나이대 분류 — 작은 CNN vs 사전학습 모델\n빨강 = 모델이 많이 본 곳 (Grad-CAM, 예측 1위 나이대 기준)")
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(sources=["upload", "webcam"], type="pil", label="사진 (업로드 / 웹캠)")
                model = gr.Radio(list(TITLES.values()), value="작은 CNN", label="모델")
                compare = gr.Checkbox(False, label="세 모델 비교")
                align = gr.Checkbox(fa.DEFAULT_ALIGN, label="얼굴 정렬 (두 눈 수평)")
                occlude = gr.Checkbox(False, label="가림 실험 (영역마다 블러 후 다시 예측)")
                btn = gr.Button("예측", variant="primary")
            with gr.Column(scale=1):
                status = gr.Markdown()
                face = gr.Image(label="잘라낸 얼굴 (모델 입력 224×224)", interactive=False, height=260)
        cols = []
        with gr.Row():
            for name in MODEL_NAMES:
                with gr.Column():
                    title = gr.Markdown(visible=False)
                    label = gr.Label(num_top_classes=3, label="예측 나이대 (상위 3개)")
                    cam = gr.Image(label="Grad-CAM", interactive=False, height=260)
                    text = gr.Markdown()
                    cols += [label, cam, title, text]
        gr.Markdown(f"{REGION_NOTICE}\n\n---\n{NOTICE}")
        btn.click(run, [inp, model, compare, align, occlude], [status, face] + cols)
    return demo


# =============================================================================
#   ▶ [모듈 4] 점검 (--selftest)
#   ① val 5장: 앱 predict = gradcam_utils 직접 계산 (확률·히트맵 같음 → 전처리 일치)
#   ② 합성 사진: val 사진을 넓은 배경(3배 캔버스, 무작위 위치)에 붙여 검출 → 예측. 원본 224 예측과 같은지
#   ③ 처리 시간: 1280×960 합성 사진 1장, 단일(모델별) / 비교 모드 × 가림 실험 끔 / 켬
# =============================================================================

def selftest(image_root):
    import pandas as pd
    from PIL import Image, ImageFilter
    val = pd.read_csv(HERE.parent / "splits_csv_v2" / "val.csv").sample(5, random_state=42)
    imgs = [Image.open(image_root / p).convert("RGB") for p in val.path]

    for name in MODEL_NAMES:
        cam = GradCAM(load_model(name, device=DEVICE), name)
        heat, _, prob = cam(preprocess(imgs))
        cam.remove()
        dp = dh = 0
        for i, im in enumerate(imgs):
            label, _, h = predict(im, name)
            dp = max(dp, np.abs(np.array([label[b] for b in BANDS]) - prob[i]).max())
            dh = max(dh, np.abs(h - heat[i]).max())
        # 5장 묶음 vs 1장 계산이라 부동소수 오차만큼은 다름 (히트맵 0~1 기준 1e-4 이하면 같은 것으로 봄)
        assert dp < 1e-5 and dh < 1e-4, (name, dp, dh)
        print(f"① {name}: val 5장 직접 계산과 비교 — 확률 최대 차이 {dp:.1e}, 히트맵 최대 차이 {dh:.1e}")

    rng = np.random.default_rng(42)
    big = pd.read_csv(HERE.parent / "splits_csv_v2" / "val.csv").sample(30, random_state=0)
    top1 = lambda f, n: int(np.argmax(list(predict(f, n)[0].values())))
    found, same, near = 0, dict.fromkeys(MODEL_NAMES, 0), dict.fromkeys(MODEL_NAMES, 0)
    for p in big.path:
        im = Image.open(image_root / p).convert("RGB")
        bg = im.resize((672, 672)).filter(ImageFilter.GaussianBlur(25))       # 사진 자체를 흐린 배경
        x, y = rng.integers(0, 448, 2)
        bg.paste(im, (int(x), int(y)))
        face, n = fa.crop_face(bg)
        if face is None:
            continue
        found += 1
        for name in MODEL_NAMES:
            d = abs(top1(face, name) - top1(im, name))
            same[name] += d == 0
            near[name] += d <= 1
    print(f"② 합성 사진(672×672 흐린 배경 + 224 얼굴) 30장: 검출 {found}장. 1위가 원본 224 예측과 같음 / ±1 나이대 안:")
    for name in MODEL_NAMES:
        print(f"   {TITLES[name]}: {same[name]}/{found}, {near[name]}/{found}")

    im = imgs[0]
    bg = im.resize((1280, 960)).filter(ImageFilter.GaussianBlur(25))
    bg.paste(im.resize((300, 300)), (600, 250))
    run(bg, "작은 CNN", True, False, True)                             # 첫 실행(워밍업) 제외
    print(f"③ 1280×960 사진 1장 처리 시간 ({DEVICE}, 검출 포함, 5번 평균)")
    for occ in (False, True):
        for title in TITLES.values():
            t = time.perf_counter()
            for _ in range(5):
                run(bg, title, False, False, occ)
            print(f"③ {title} 단일, 가림 실험 {'켬' if occ else '끔'}: {(time.perf_counter() - t) / 5:.3f}초")
        t = time.perf_counter()
        for _ in range(5):
            run(bg, "작은 CNN", True, False, occ)
        print(f"③ 세 모델 비교, 가림 실험 {'켬' if occ else '끔'}: {(time.perf_counter() - t) / 5:.3f}초")
    t = time.perf_counter()
    for _ in range(5):
        fa.crop_face(bg)
    print(f"   (그중 얼굴 검출·자르기만: {(time.perf_counter() - t) / 5:.3f}초)")


# =============================================================================
#   ▶ [모듈 5] 공유 범위(이 PC / 같은 네트워크 / 공유 링크)와 로그인
#   - 로컬 IP: UDP 소켓을 바깥 주소로 "연결"만 해서(실제로 보내는 데이터 없음) 이 PC가 쓰는 네트워크 주소를 알아냄
# =============================================================================

def local_ip():
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


def launch_options(a):
    """명령행 옵션 → demo.launch 인자. 경고와 접속 주소 안내도 여기서 출력"""
    if (a.share or a.lan) and not a.auth:
        who = "인터넷의 누구나" if a.share else "같은 네트워크의 누구나"
        print(f"[경고] --auth 없이 공유합니다. 주소를 아는 {who} 접속해 사진을 올릴 수 있습니다. "
              f"--auth 아이디:비번 을 함께 쓰세요.", flush=True)
    if a.lan:
        print(f"[같은 네트워크 접속 주소] http://{local_ip()}:{a.port}  "
              f"(Windows 방화벽 창이 뜨면 '개인 네트워크' 허용)", flush=True)
    return dict(server_name="0.0.0.0" if a.lan else "127.0.0.1", server_port=a.port, share=a.share,
                auth=tuple(a.auth.split(":", 1)) if a.auth else None,
                auth_message="인공지능 4조 얼굴 나이대 분류 시연 — 팀에서 받은 아이디·비밀번호를 입력하세요")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--image-root", type=Path)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Gradio 공유 링크(gradio.live) 생성")
    ap.add_argument("--lan", action="store_true", help="같은 네트워크에서 접속 허용 (0.0.0.0)")
    ap.add_argument("--auth", metavar="아이디:비번", help="로그인 걸기")
    a = ap.parse_args()
    if a.auth and (":" not in a.auth or not all(a.auth.split(":", 1))):
        ap.error("--auth는 아이디:비번 형식이어야 합니다 (예: --auth team4:비밀번호)")
    if a.selftest:
        selftest(a.image_root)
    else:
        demo = build()
        _, _, share_url = demo.launch(**launch_options(a), prevent_thread_lock=True)
        if a.share and not share_url:
            print("[공유 링크 실패] gradio.live 서버에 연결하지 못했습니다(인터넷·Gradio 서버 상태). "
                  "앱은 이 PC에서만 열려 있습니다. 같은 네트워크 팀원에게는 --lan 을 쓰세요.", flush=True)
        demo.block_thread()                     # Ctrl+C → 서버 닫고 정상 종료 → atexit로 임시 폴더 삭제
