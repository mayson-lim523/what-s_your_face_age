# #############################################################################
#
#   얼굴 나이대 분류 — Gradio 시연 화면
#   인공지능 4조 · 시연 화면 담당: LIM
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
    "small_cnn": os.path.join(HERE, "weights", "small_cnn_best.pth"),        # 작은CNN 학습 코드의 small_cnn_best.pth
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
    p.add_argument("--margin", type=float, default=0.15)   # 얼굴 상자 여백 (상자 크기 대비 비율, 위·아래·양옆 각각)
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
#   여백 기본값 0.15: 학습 사진(224×224)에서 Haar 상자 너비가 사진의 약 79%(중앙값)
#                    → 상자 + 양쪽 15% = 1.3배로 자르면 상자 비율 약 77%로 비슷해짐
#   얼굴로 꽉 찬 사진 대책 (테스트 사진 500장 기준 검출률 58% → 90%):
#     ① 검출 전에 사진 둘레에 검은 테두리를 붙임 — Haar는 얼굴 주변 여유 공간이 있어야 잘 찾음 (58% → 83%)
#     ② 기본 cascade가 못 찾으면 alt2 → 옆얼굴 → 옆얼굴(좌우 반전) 순서로 다시 시도 (83% → 90%)
# =============================================================================

PAD_RATIO = 0.25                                        # 테두리 두께 (사진 긴 변 대비 비율)
_H = cv2.data.haarcascades                              # OpenCV에 같이 들어 있는 cascade 파일 폴더 (추가 다운로드 없음)
CASCADES = [                                            # (검출기, 좌우 반전 여부) — 위에서부터 차례로 시도
    (cv2.CascadeClassifier(_H + "haarcascade_frontalface_default.xml"), False),   # 정면 얼굴 (기본)
    (cv2.CascadeClassifier(_H + "haarcascade_frontalface_alt2.xml"), False),      # 정면 얼굴 (다른 학습본)
    (cv2.CascadeClassifier(_H + "haarcascade_profileface.xml"), False),           # 옆얼굴 (왼쪽을 보는 얼굴만 학습됨)
    (cv2.CascadeClassifier(_H + "haarcascade_profileface.xml"), True),            # 좌우 반전 → 오른쪽을 보는 옆얼굴
]


# TODO: 팀 전처리와 같은 검출·정렬 방법으로 교체
def detect_and_crop(image, margin=0.15):                # (image=PIL 이미지, margin=상자 크기 대비 여백 비율)
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
    x, y = x - p, y - p                                 # 테두리 붙인 좌표 → 원본 사진 좌표
    dx, dy = int(w * margin), int(h * margin)           # 여백 픽셀 수 (가로, 세로)
    left, top = max(x - dx, 0), max(y - dy, 0)          # 여백만큼 넓히되 사진 밖으로 나가지 않게
    right, bottom = min(x + w + dx, W), min(y + h + dy, H)
    return image.convert("RGB").crop((left, top, right, bottom))   # (왼, 위, 오, 아래) 영역 잘라내기

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

def build_ui(models, first, device, margin):            # (models={이름: (모델, GradCAM)}, first=처음 선택 모델, device=장치 문자열, margin=얼굴 여백) → gr.Blocks 화면
    def run(img, name):                                 # (img=PIL 이미지 또는 None, name=드롭다운에서 고른 모델) 버튼을 누르면 실행
        if img is None:                                 # 사진 없이 버튼을 누른 경우
            raise gr.Error("사진을 먼저 올리거나 촬영해 주세요")   # 화면에 빨간 알림
        model, cam = models[name]                       # 고른 모델과 그 모델에 걸린 GradCAM
        face = detect_and_crop(img, margin)             # 얼굴 잘라내기
        if face is None:                                # 얼굴 못 찾음
            gr.Warning(NO_FACE_MSG)                     # 화면에 노란 알림
            return None, None, None                     # 이전 결과가 남지 않게 출력 비우기
        probs = predict(model, face, device)            # {나이대: 확률}
        heat, _ = cam(val_transform(face).unsqueeze(0).to(device))   # (224, 224) 예측 1위 나이대의 히트맵
        # ponytail: GradCAM은 hook 결과를 객체에 저장하므로 요청을 한 번에 하나씩 처리해야 함 (Gradio 기본 concurrency_limit=1이라 안전)
        return probs, face, overlay(face, heat)         # (확률, 잘라낸 얼굴, 겹친 이미지)

    # delete_cache=(주기 초, 기준 초): 10분마다 확인해서 만든 지 10분 넘은 임시 파일 삭제 (서버를 끌 때도 삭제)
    with gr.Blocks(title="얼굴 나이대 분류", delete_cache=(600, 600)) as demo:
        gr.Markdown("## 얼굴 나이대 분류 시연\n사진은 저장되지 않습니다.")   # 제목·안내
        pick = gr.Dropdown(choices=list(models), value=first, label="모델")   # (choices=가중치가 있는 모델만, value=처음 선택값)
        with gr.Row():                                  # 가로로 나란히
            inp = gr.Image(sources=["upload", "webcam"], type="pil", label="얼굴 사진")   # (sources=입력 방법, type=PIL로 받기)
            out_face = gr.Image(type="pil", label="잘라낸 얼굴")                         # 모델에 들어간 얼굴 영역
            out_cam = gr.Image(type="pil", label="Grad-CAM (1위 나이대의 판단 근거, 빨강 = 많이 본 곳)")   # 히트맵 겹친 이미지
            out_label = gr.Label(num_top_classes=3, label="예측 나이대 (상위 3개)")      # (num_top_classes=보여줄 개수)
        btn = gr.Button("예측하기", variant="primary")  # (variant=강조 색) 실행 버튼
        btn.click(run, inputs=[inp, pick], outputs=[out_label, out_face, out_cam])   # 버튼 → run(사진, 모델) → (확률, 얼굴, 히트맵)
    return demo                                         # 화면 객체

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

    if args.image:                                      # 한 장 점검 모드 (화면과 같은 순서: 검출 → 예측)
        face = detect_and_crop(Image.open(args.image), args.margin)   # 얼굴 잘라내기
        if face is None:                                # 못 찾으면 안내만 출력
            print(NO_FACE_MSG)
        else:
            result = predict(model, face, args.device)  # 확률 7개
            top3 = sorted(result.items(), key=lambda kv: kv[1], reverse=True)[:3]   # 확률 큰 순으로 3개
            print(f"  얼굴 크기 {face.size}")           # (너비, 높이) 잘라낸 크기
            for name, p in top3:                        # 한 줄씩 출력
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
