# #############################################################################
#
#   얼굴 나이대 분류 — Gradio 시연 화면 [주변 Blur 전용]
#   인공지능 4조 · 시연 화면 담당: LIM
#
#   [수정 범위] 원본 app.py의 G-1, G-2, G-4(학습 모델·예측)는 유지합니다.
#   [추가] G-3: 대상 얼굴 하나 주변만 흐리게 만드는 전처리 함수
#   [수정] G-5: 주변 Blur를 적용한 단일 입력의 예측과 Grad-CAM만 출력
#   [미포함] 다중 얼굴 개별 예측 / 프레임별 추적 / 실시간 스트리밍
#   [참고] GitHub: https://github.com/Pooja-Vachhad/Custom-Face-Blur-Overlay-Filter/blob/main/main.py
#           얼굴 영역을 구분해 GaussianBlur를 적용하는 아이디어 참고 (코드 전체 복사 아님)
#   [주의] 이 파일은 Blur 처리된 이미지만 CNN에 입력합니다. 전후 비교 및 다중 얼굴 분석은 없습니다.
#
#   [모듈 구성]
#     G-1 설정·명령줄 옵션   G-2 모델 불러오기   G-3 얼굴 검출·자르기
#     G-4 전처리·예측        G-5 화면 (모델 드롭다운 + Grad-CAM)
#
#   [실행 방법] 저장소 최상위 폴더에서
#     화면 실행  : python 시연\app.py            → 브라우저에서 http://127.0.0.1:7860
#     한 장 점검 : python 시연\app.py --image 사진.jpg
#     다른 가중치: python 시연\app.py --model small_cnn --weights 경로\small_cnn_best.pth
#   드롭다운에는 시연\weights\ 에 가중치 파일이 있는 모델만 나옴 (DEFAULT_WEIGHTS 참고)
#
# #############################################################################


# =============================================================================
#   ▶ [모듈 G-1 시작] 설정·명령줄 옵션
#   출력: 상수들, parse_args()
# =============================================================================

import argparse                                         # 명령줄 옵션(--model 등) 읽기
import os                                               # 경로 합치기
import sys                                              # import 경로(sys.path)에 작은CNN 폴더 추가

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")   # Gradio 사용 통계 전송 끔 (gradio import 전에 설정해야 적용)
import cv2                                              # OpenCV: 얼굴 검출 (Haar cascade, 4.x 버전 필요)
import gradio as gr                                     # 웹 시연 화면
import numpy as np                                      # PIL ↔ OpenCV 배열 변환
import timm                                             # 팀원 모델(ResNet-50, ConvNeXt-Tiny) 구조 만들기
import torch                                            # PyTorch 본체
from PIL import Image                                   # 사진 다루기

HERE = os.path.dirname(os.path.abspath(__file__))       # 이 파일이 있는 폴더 (시연/)
ROOT = os.path.dirname(HERE)                            # 저장소 최상위 폴더
sys.path.insert(0, os.path.join(ROOT, "작은CNN"))       # 작은CNN 폴더의 파일을 import 할 수 있게 경로 추가
# SmallCNN 구조와 검증용 transform을 학습 코드에서 그대로 가져옴 (학습 코드는 __main__ 안이라 실행되지 않음)
from CNN_4layers_unified import SmallCNN, val_transform   # val_transform: Resize 224×224 → ToTensor → Normalize(MEAN, STD)
from gradcam import GradCAM, overlay, target_layer     # 같은 폴더의 Grad-CAM 모듈

BAND_NAMES = ["0-9세", "10대", "20대", "30대", "40대", "50대", "60세 이상"]   # 나이대 이름 (eval_small_cnn.py와 같은 순서, 번호 0~6)
DEFAULT_WEIGHTS = {                                     # 모델별 기본 가중치 경로 — 재학습 후에는 이 파일만 바꿔 끼우면 됨
    # 작은CNN v2 가중치 (splits_csv_v2 학습, 17 에폭, 검증 macro-F1 0.461, 개발일지 14). 다른 가중치는 --weights 로
    "small_cnn": os.path.join(ROOT, "작은CNN", "results", "small_cnn_v2_lr1e-3", "small_cnn_best.pth"),
    "resnet50": os.path.join(HERE, "weights", "resnet50_best.pt"),           # ResNet50 step4_train.py의 best.pt 이름만 바꿔 복사
    "convnext_tiny": os.path.join(HERE, "weights", "convnext_tiny_best.pt"), # ConvNeXt step4_train.py의 best.pt 이름만 바꿔 복사
}
NO_FACE_MSG = "얼굴을 찾지 못했습니다. 정면 사진으로 다시 시도해 주세요"   # 얼굴 검출 실패 안내 문구


def parse_args():                                       # 명령줄 옵션을 읽어서 돌려줌
    p = argparse.ArgumentParser(description="얼굴 나이대 분류 시연")   # (description=도움말 첫 줄)
    p.add_argument("--model", default="small_cnn", choices=list(DEFAULT_WEIGHTS))   # (choices=고를 수 있는 값) 처음 선택될 모델
    p.add_argument("--weights", default=None)           # --model의 가중치 파일 경로. 안 주면 DEFAULT_WEIGHTS[모델]
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")   # GPU가 없으면 CPU
    p.add_argument("--port", type=int, default=7860)    # 브라우저 주소의 포트 번호
    p.add_argument("--margin", type=float, nargs=4, default=MARGINS,   # (nargs=4: 숫자 4개) 얼굴 상자 여백 왼 위 오른 아래 (상자 크기 대비 비율)
                   metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    p.add_argument("--image", default=None)             # 사진 경로를 주면 화면 없이 상위 3개만 출력하고 끝 (점검용)
    return p.parse_args()                               # args.model, args.weights ... 로 사용

# -----------------------------------------------------------------------------
# ◀ [모듈 G-1 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   ▶ [모듈 G-2 시작] 모델 불러오기
#   입력: 모델 이름, 가중치 경로, 장치
#   출력: eval() 상태의 모델
# =============================================================================

def load_model(name, weights, device):                  # (name=모델 이름, weights=가중치 경로, device=장치 문자열)
    if name == "small_cnn":
        model = SmallCNN()                              # 빈 모델(무작위 가중치) 생성
    else:                                               # 팀원 모델: common.py build_model과 같은 방법 (사전학습 가중치는 안 받음)
        model = timm.create_model(name, pretrained=False, num_classes=len(BAND_NAMES))   # (이름, pretrained=다운로드 여부, num_classes=출력 7)
    # torch.load(경로, map_location=장치): 저장된 가중치 딕셔너리를 읽어 지정 장치에 올림 (GPU에서 저장한 것을 CPU에서도 열 수 있게)
    ckpt = torch.load(weights, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt  # 팀원 best.pt: {"model": state_dict, "model_name", "epoch"} / SmallCNN: state_dict만
    model.load_state_dict(state)                        # 층 이름이 같은 자리에 가중치를 채움 (하나라도 다르면 에러)
    return model.to(device).eval()                      # 장치로 옮기고 평가 모드 (Dropout 끔, BN은 저장된 통계 사용)

# -----------------------------------------------------------------------------
# ◀ [모듈 G-2 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   ▶ [모듈 G-3 시작] 얼굴 검출·자르기
#   입력: PIL 이미지 (RGB)
#   출력: 잘라낸 얼굴 PIL 이미지, 못 찾으면 None
#   여백 MARGINS: v2 학습 사진(division_224AFAD-X/training) 나이대별 300장 = 2,100장에 아래 detect_face를 돌려 잰 값
#                 (검출 1,795장, 중앙값): 상자 폭 = 사진 폭의 0.96배, 상자 중심 (x 0.49, y 0.42)
#                 상자 밖 여백 (상자 크기 대비) 왼 0.014, 위 −0.065, 오른 0.028, 아래 0.105 — 나이대별 차이 ±0.01
#                 → 왼·오른은 평균 0.02로 같게, 위는 상자 안쪽에서 자름(음수), 아래는 상자보다 더 내려감
#                 (예전 값: 사방 0.15 — 예전 데이터에서 상자 폭이 사진의 79%였을 때 기준)
#   얼굴로 꽉 찬 사진 대책 (테스트 사진 500장 기준 검출률 58% → 90%):
#     ① 검출 전에 사진 둘레에 검은 테두리를 붙임 — Haar는 얼굴 주변 여유 공간이 있어야 잘 찾음 (58% → 83%)
#     ② 기본 cascade가 못 찾으면 alt2 → 옆얼굴 → 옆얼굴(좌우 반전) 순서로 다시 시도 (83% → 90%)
# =============================================================================

MARGINS = (0.02, -0.065, 0.02, 0.105)                   # 상자 크기 대비 여백 (왼, 위, 오른, 아래). 음수 = 상자 안쪽으로 자름
# [Blur 추가 G-3] 선택된 얼굴 주변도 포함해 잘라내고 주변 영역만 흐림 처리합니다.
CONTEXT_RATIO = 0.50                                # 얼굴 크기의 50%만큼 상하좌우로 주변 영역 확보
BLUR_KERNEL = (31, 31)                              # GaussianBlur는 홀수 크기의 커널 사용
PAD_RATIO = 0.25                                        # 테두리 두께 (사진 긴 변 대비 비율)
_H = cv2.data.haarcascades                              # OpenCV에 같이 들어 있는 cascade 파일 폴더 (추가 다운로드 없음)
CASCADES = [                                            # (검출기, 좌우 반전 여부) — 위에서부터 차례로 시도
    (cv2.CascadeClassifier(_H + "haarcascade_frontalface_default.xml"), False),   # 정면 얼굴 (기본)
    (cv2.CascadeClassifier(_H + "haarcascade_frontalface_alt2.xml"), False),      # 정면 얼굴 (다른 학습본)
    (cv2.CascadeClassifier(_H + "haarcascade_profileface.xml"), False),           # 옆얼굴 (왼쪽을 보는 얼굴만 학습됨)
    (cv2.CascadeClassifier(_H + "haarcascade_profileface.xml"), True),            # 좌우 반전 → 오른쪽을 보는 옆얼굴
]


# TODO: 팀 전처리와 같은 검출·정렬 방법으로 교체
def detect_face(image):                                 # (image=PIL 이미지) → 가장 큰 얼굴 상자 (x, y, w, h) 원본 좌표, 못 찾으면 None
    rgb = np.array(image.convert("RGB"))                # PIL → (H, W, 3) 배열, RGB 순서
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)        # Haar는 흑백으로 검출 (H, W)
    H, W = gray.shape                                   # 원본 사진 높이, 너비
    p = int(max(H, W) * PAD_RATIO)                      # 테두리 두께 (픽셀)
    padded = cv2.copyMakeBorder(gray, p, p, p, p, cv2.BORDER_CONSTANT, value=0)   # (위, 아래, 왼, 오 두께, 검은색) → (H+2p, W+2p)
    boxes = []                                          # 찾은 얼굴 상자들 (x, y, w, h)
    for cascade, flip in CASCADES:                      # 검출기를 차례로 시도
        g = cv2.flip(padded, 1) if flip else padded     # (사진, 1=좌우 반전)
        # (흑백 사진, scaleFactor=검색 창을 1.1배씩 키움, minNeighbors=겹친 후보가 5개 이상이어야 얼굴로 인정, minSize=최소 얼굴 크기)
        boxes = cascade.detectMultiScale(g, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(boxes) > 0:                              # 찾았으면 더 시도하지 않음
            if flip:                                    # 반전 사진에서 찾은 상자 → 원래 방향 좌표로 (x를 거울처럼)
                boxes = [(g.shape[1] - x - w, y, w, h) for x, y, w, h in boxes]
            break
    if len(boxes) == 0:                                 # 모든 검출기가 못 찾음
        return None
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])  # 여러 얼굴이면 넓이(w×h)가 가장 큰 얼굴
    return int(x - p), int(y - p), int(w), int(h)       # 테두리 붙인 좌표 → 원본 사진 좌표 (사진 밖으로 나갈 수 있음)


def detect_and_crop(image, margin=MARGINS):             # (image=PIL 이미지, margin=(왼, 위, 오른, 아래) 상자 크기 대비 여백 비율)
    box = detect_face(image)                            # 얼굴 상자 (x, y, w, h) 또는 None
    if box is None:
        return None
    x, y, w, h = box
    W, H = image.size                                   # 원본 사진 너비, 높이
    ml, mt, mr, mb = margin                             # 여백 4개
    left, top = max(x - int(w * ml), 0), max(y - int(h * mt), 0)   # 여백만큼 넓히되(음수면 좁히되) 사진 밖으로 나가지 않게
    right, bottom = min(x + w + int(w * mr), W), min(y + h + int(h * mb), H)
    return image.convert("RGB").crop((left, top, right, bottom))   # (왼, 위, 오, 아래) 영역 잘라내기



# =============================================================================
#   [Blur 추가 G-3] 원본 detect_face()/detect_and_crop()는 유지합니다.
#   기존 코드처럼 가장 큰 얼굴 1개만 대상으로 선택합니다.
#   이 함수는 선택한 얼굴을 선명하게 남기고 주변만 Blur 처리한 이미지 1개를 반환합니다.
#   전후 비교용 원본 입력은 만들지 않습니다.
# =============================================================================

def create_blurred_face_input(image, box, context_ratio=CONTEXT_RATIO):
    # [Blur 추가 G-3] 얼굴과 그 주변이 함께 보이도록 원본에서 영역을 잘라냅니다.
    image = image.convert("RGB")
    W, H = image.size
    x, y, w, h = box
    left = max(0, x - int(w * context_ratio))
    top = max(0, y - int(h * context_ratio))
    right = min(W, x + w + int(w * context_ratio))
    bottom = min(H, y + h + int(h * context_ratio))
    if right <= left or bottom <= top:
        return None

    roi = np.array(image.crop((left, top, right, bottom)))

    # [Blur 추가 G-3] 선택한 얼굴을 포함한 주변 영역 전체에 Gaussian Blur를 적용합니다.
    blurred = cv2.GaussianBlur(roi, BLUR_KERNEL, 0)

    # [Blur 추가 G-3] 선택된 얼굴 상자 영역만 원본 픽셀로 덮어씁니다.
    # 따라서 얼굴 1개는 선명하게 유지하고, 주변 얼굴·배경은 흐리게 보입니다.
    fx1 = max(0, x - left)
    fy1 = max(0, y - top)
    fx2 = min(roi.shape[1], x + w - left)
    fy2 = min(roi.shape[0], y + h - top)
    if fx2 > fx1 and fy2 > fy1:
        blurred[fy1:fy2, fx1:fx2] = roi[fy1:fy2, fx1:fx2]

    # [Blur 추가 G-3] 단일 Blur 이미지만 반환합니다. 기존 val_transform이 224x224로 맞춥니다.
    return Image.fromarray(blurred)

# -----------------------------------------------------------------------------
# ◀ [모듈 G-3 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   ▶ [모듈 G-4 시작] 전처리·예측
#   입력: PIL 얼굴 이미지 (RGB)
#   출력: {나이대 이름: 확률} 딕셔너리 7개 (gr.Label이 바로 받는 형식)
# =============================================================================

def predict(model, img, device):                        # (model=모델, img=PIL 이미지, device=장치 문자열)
    x = val_transform(img.convert("RGB")).unsqueeze(0).to(device)   # (3, 224, 224) → (1, 3, 224, 224) 배치 차원 추가
    with torch.no_grad():                               # 예측만 하므로 기울기 기록 끔
        logits = model(x)                               # (1, 7) 점수
    probs = torch.softmax(logits, dim=1)[0].cpu()       # (7,) 확률 (합 1)
    return {BAND_NAMES[i]: float(probs[i]) for i in range(len(BAND_NAMES))}   # {"20대": 0.53, ...}

# -----------------------------------------------------------------------------
# ◀ [모듈 G-4 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   ▶ [모듈 G-5 시작] 화면
#   입력: 업로드 사진 또는 웹캠 촬영 사진
#   출력: 상위 3개 나이대와 확률, 잘라낸 얼굴, Grad-CAM 겹친 이미지
#   개인정보: 사진은 메모리(PIL)로만 다루고 직접 저장하지 않음.
#            Gradio가 내부적으로 만드는 임시 파일은 delete_cache로 자동 삭제
# =============================================================================

def build_ui(models, first, device, margin):            # [기존 G-5 유지] Gradio 모델 드롭다운 등 재사용
    def run(img, name):
        if img is None:
            raise gr.Error("사진을 먼저 올리거나 촬영해 주세요")
        model, cam = models[name]                        # [기존 G-5 유지] 기존 모델과 Grad-CAM 사용
        box = detect_face(img)                           # [기존 G-3 유지] 가장 큰 얼굴 1개만 선택
        if box is None:
            gr.Warning(NO_FACE_MSG)
            return None, None, None

        # [Blur 수정 G-5] Blur를 적용한 입력 이미지 하나만 생성합니다.
        blurred_face = create_blurred_face_input(img, box)
        if blurred_face is None:
            gr.Warning("선택한 얼굴의 입력 영역을 만들 수 없습니다.")
            return None, None, None

        # [Blur 수정 G-5] 기존 CNN 예측 함수에 Blur 처리된 이미지 하나만 입력합니다.
        probs = predict(model, blurred_face, device)

        # [Blur 수정 G-5] 기존 Grad-CAM으로 Blur 입력의 중요 영역을 한 번만 시각화합니다.
        # Grad-CAM은 기울기가 필요하므로 torch.no_grad()로 감싸지 않습니다.
        heat, _ = cam(val_transform(blurred_face).unsqueeze(0).to(device))
        return probs, blurred_face, overlay(blurred_face, heat)

    # [기존 G-5 유지] 원본과 동일한 3개 출력(확률, 입력 이미지, Grad-CAM)을 사용합니다.
    with gr.Blocks(title="얼굴 나이대 분류 — 주변 Blur", delete_cache=(600, 600)) as demo:
        gr.Markdown("## 얼굴 나이대 분류 — 주변 Blur 적용\n사진은 저장되지 않습니다.")
        pick = gr.Dropdown(choices=list(models), value=first, label="모델")
        with gr.Row():
            inp = gr.Image(sources=["upload", "webcam"], type="pil", label="얼굴 사진")
            out_face = gr.Image(type="pil", label="주변 Blur 처리된 입력 이미지")
            out_cam = gr.Image(type="pil", label="Grad-CAM (Blur 입력 기준)")
            out_label = gr.Label(num_top_classes=3, label="예측 나이대 (상위 3개)")
        btn = gr.Button("예측하기", variant="primary")
        btn.click(run, inputs=[inp, pick], outputs=[out_label, out_face, out_cam])
    return demo

# -----------------------------------------------------------------------------
# ◀ [모듈 G-5 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   실행부
# =============================================================================
if __name__ == "__main__":
    args = parse_args()                                 # 명령줄 옵션
    weights = args.weights or DEFAULT_WEIGHTS[args.model]   # 가중치 경로 (옵션이 없으면 기본값)
    model = load_model(args.model, weights, args.device)    # 모델 준비
    print(f"모델 {args.model} | 가중치 {weights} | 장치 {args.device}")

    if args.image:                                      # [Blur 수정 실행부] Blur 적용 이미지 하나만 예측
        image = Image.open(args.image).convert("RGB")
        box = detect_face(image)
        if box is None:
            print(NO_FACE_MSG)
        else:
            blurred_face = create_blurred_face_input(image, box)
            if blurred_face is None:
                print("선택한 얼굴의 입력 영역을 만들 수 없습니다.")
            else:
                result = predict(model, blurred_face, args.device)
                top3 = sorted(result.items(), key=lambda kv: kv[1], reverse=True)[:3]
                print(f"  Blur 적용 이미지 크기 {blurred_face.size}")
                for name, p in top3:
                    print(f"  {name:8s} {p:.3f}")
    else:                                               # 화면 모드
        # (server_name=127.0.0.1: 이 PC에서만 접속, share=False: 외부 공유 링크 안 만듦)
        paths = {**DEFAULT_WEIGHTS, args.model: weights}   # 모델별 가중치 경로 (--weights 를 준 모델은 그 경로로)
        models = {}                                     # {이름: (모델, GradCAM)} — 가중치 파일이 있는 모델만
        for name, path in paths.items():
            if not os.path.exists(path):                # 가중치가 없으면 목록에서 뺌
                print(f"  {name}: 가중치 없음 ({path}) → 목록에서 제외")
                continue
            m = model if name == args.model else load_model(name, path, args.device)   # --model은 위에서 이미 불러옴
            models[name] = (m, GradCAM(m, *target_layer(m, name)))   # 대상 층에 hook 걸기 (*: (층, use_input) 튜플을 풀어서 전달)
        build_ui(models, args.model, args.device, args.margin).launch(server_name="127.0.0.1", server_port=args.port, share=False)

# #############################################################################
# 모듈 끝: app.py
# #############################################################################