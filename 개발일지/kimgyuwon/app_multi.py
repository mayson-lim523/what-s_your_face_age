# #############################################################################
#
#   얼굴 나이대 분류 — Gradio 시연 화면 [다중 얼굴 전용]
#   인공지능 4조 · 시연 화면 담당: LIM
#
#   [수정 범위] 원본 app.py의 G-1, G-2, G-4(학습 모델·예측)는 유지합니다.
#   [추가] G-3: 검출된 모든 얼굴의 좌표·Crop을 반환
#   [수정] G-5: 얼굴마다 나이대 예측과 Grad-CAM을 각각 생성해 갤러리에 표시
#   [미포함] 주변 Blur / 프레임별 추적 / 실시간 스트리밍
#   [참고] GitHub: https://github.com/samueladebisi-dev/gender-age-detection/blob/main/detect.py
#           검출 얼굴마다 순회하여 나이 분류하는 구조를 참고 (코드 전체 복사 아님)
#   [주의] webcam 입력은 사진 촬영이며, 영상의 연속 프레임 처리가 아닙니다.
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
#   [다중 추가 G-3] 원본 detect_face()/detect_and_crop()는 수정하지 않고 보존.
#   아래 함수들은 "가장 큰 얼굴 1개" 대신 "검출된 얼굴 N개"를 처리하기 위해 추가됨.
# =============================================================================

def _face_box_iou(a, b):
    # [다중 추가 G-3] 여러 Haar 검출기가 같은 얼굴을 찾았을 때 중복 제거할 기준 계산.
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    overlap = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - overlap
    return overlap / union if union > 0 else 0.0


def detect_faces_all(image):
    # [다중 추가 G-3] 기존 Haar Cascade와 검은 테두리 방식은 그대로 사용.
    # [다중 변경 G-3] 첫 검출에서 break하지 않고 4종 Cascade 결과를 모음.
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    p = int(max(H, W) * PAD_RATIO)
    padded = cv2.copyMakeBorder(gray, p, p, p, p, cv2.BORDER_CONSTANT, value=0)
    candidates = []

    for cascade, flip in CASCADES:
        if cascade.empty():
            continue
        g = cv2.flip(padded, 1) if flip else padded
        boxes = cascade.detectMultiScale(
            g, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        for x, y, w, h in boxes:
            if flip:
                x = g.shape[1] - x - w
            # [다중 추가 G-3] 테두리 기준 좌표를 원본 사진 좌표로 복원하고 범위를 제한.
            left = max(0, int(x - p))
            top = max(0, int(y - p))
            right = min(W, int(x - p + w))
            bottom = min(H, int(y - p + h))
            if right > left and bottom > top:
                candidates.append((left, top, right - left, bottom - top))

    # [다중 추가 G-3] 같은 얼굴을 여러 번 표시하지 않도록 겹친 박스 하나만 유지.
    chosen = []
    for box in sorted(candidates, key=lambda b: b[2] * b[3], reverse=True):
        if all(_face_box_iou(box, prev) < 0.35 for prev in chosen):
            chosen.append(box)

    # [다중 추가 G-3] 정지 사진에서 번호가 일정하도록 왼쪽→오른쪽 순서 정렬.
    # [주의] 얼굴 1번이 "먼저 들어온 사람"이라는 의미는 아님.
    return sorted(chosen, key=lambda b: (b[0] + b[2] / 2, b[1]))


def detect_and_crop_all(image, margin=MARGINS):
    # [다중 추가 G-3] 모델에는 얼굴별 Crop 이미지를 각각 전달.
    # 반환 형식: [(PIL 얼굴 이미지, (x,y,w,h)), ...]
    W, H = image.size
    ml, mt, mr, mb = margin
    result = []
    for x, y, w, h in detect_faces_all(image):
        left = max(0, x - int(w * ml))
        top = max(0, y - int(h * mt))
        right = min(W, x + w + int(w * mr))
        bottom = min(H, y + h + int(h * mb))
        if right <= left or bottom <= top:
            continue
        face = image.convert("RGB").crop((left, top, right, bottom))
        result.append((face, (x, y, w, h)))
    return result


def draw_face_boxes(image, rows):
    # [다중 추가 G-5에서 호출] 원본 사진에 얼굴별 번호와 검출 박스 표시.
    rgb = np.array(image.convert("RGB")).copy()
    for index, (_, box, _) in enumerate(rows, start=1):
        x, y, w, h = box
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(rgb, f"Face {index}", (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 0), 2)
    return Image.fromarray(rgb)

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

def build_ui(models, first, device, margin):            # [기존 G-5 유지] Gradio 입력 및 모델 선택
    def run(img, name):
        if img is None:
            raise gr.Error("사진을 먼저 올리거나 촬영해 주세요")
        model, cam = models[name]                       # [기존 G-5 유지] 모델·Grad-CAM 객체 선택
        faces = detect_and_crop_all(img, margin)        # [다중 수정 G-5] 얼굴 1개 → 검출된 모든 얼굴 가져오기
        if not faces:
            gr.Warning(NO_FACE_MSG)
            return None, [], [], "얼굴을 검출하지 못했습니다."

        face_gallery = []
        cam_gallery = []
        predictions = []
        rows = []

        for index, (face, box) in enumerate(faces, start=1):
            # [다중 추가 G-5] 얼굴마다 기존 predict()/cam()을 한 번씩 호출.
            # [다중 추가 G-5] 다른 사람 얼굴과 특징이 섞이지 않도록 Crop별로 예측.
            probs = predict(model, face, device)
            heat, _ = cam(val_transform(face).unsqueeze(0).to(device))
            best_class = max(probs, key=probs.get)
            best_probability = probs[best_class]
            label = f"Face {index}: {best_class} ({best_probability:.1%})"
            face_gallery.append((face, label))
            cam_gallery.append((overlay(face, heat), label))
            predictions.append(f"{label}")
            rows.append((face, box, probs))

        # [다중 추가 G-5] 이미지별 얼굴 번호, 얼굴별 Crop, 얼굴별 Grad-CAM을 동시 출력.
        boxed_image = draw_face_boxes(img, rows)
        summary = f"### 검출된 얼굴: {len(rows)}개\n" + "\n\n".join(predictions)
        return boxed_image, face_gallery, cam_gallery, summary

    # [기존 G-5 유지] Gradio Blocks/Dropdown/Image 입력 방식 유지.
    with gr.Blocks(title="얼굴 나이대 분류 — 다중 얼굴", delete_cache=(600, 600)) as demo:
        gr.Markdown("## 얼굴 나이대 분류 — 다중 얼굴 개별 분석\n사진은 저장되지 않습니다.")
        pick = gr.Dropdown(choices=list(models), value=first, label="모델")
        with gr.Row():
            inp = gr.Image(sources=["upload", "webcam"], type="pil", label="얼굴 사진")
            # [다중 추가 G-5] 한 사진에 N개 얼굴이 어디서 검출됐는지 표시.
            out_boxes = gr.Image(type="pil", label="모든 얼굴 검출 위치")
        with gr.Row():
            # [다중 수정 G-5] 원본의 단일 gr.Image 출력을 N개 항목의 gr.Gallery로 교체.
            out_faces = gr.Gallery(label="얼굴별 Crop", columns=3)
            out_cams = gr.Gallery(label="얼굴별 Grad-CAM", columns=3)
        # [다중 수정 G-5] 단일 gr.Label 대신 얼굴마다 1위 나이대·확률 표시.
        out_summary = gr.Markdown(label="얼굴별 예측 결과")
        btn = gr.Button("예측하기", variant="primary")
        btn.click(run, inputs=[inp, pick], outputs=[out_boxes, out_faces, out_cams, out_summary])
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

    if args.image:                                      # [다중 수정 실행부] CMD 모드도 모든 얼굴에 대해 예측
        image = Image.open(args.image).convert("RGB")
        faces = detect_and_crop_all(image, args.margin)
        if not faces:
            print(NO_FACE_MSG)
        for index, (face, box) in enumerate(faces, start=1):
            result = predict(model, face, args.device)
            top3 = sorted(result.items(), key=lambda kv: kv[1], reverse=True)[:3]
            print(f"  Face {index} | 좌표 {box} | 얼굴 크기 {face.size}")
            for name, p in top3:
                print(f"    {name:8s} {p:.3f}")
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