# #############################################################################
#
#   얼굴 검출 · 눈 수평 정렬 · 자르기 — 시연 앱(app_gradio.py)과 여백 측정(face_crop_measure.py) 공용
#   인공지능 4조 · 담당: LIM
#
#   [왜 필요한가]
#     학습 사진(division_224AFAD-X)은 얼굴이 이미 꽉 차게 잘린 224 사진이다. 저장소에는 그 사진을 자른
#     코드가 없어서(팀원 코드는 분류·분할만 함), YuNet 상자 기준 여백을 학습 사진에서 거꾸로 재서 맞춘다.
#
#   [순서]
#     ① 사진 둘레에 테두리(가장자리 픽셀 복제) → 얼굴이 사진 끝까지 찬 사진도 검출되게.
#        검은색 대신 복제로 채워서, 자를 범위가 사진 밖으로 나가도 검은 띠가 생기지 않게 함
#     ② YuNet 검출 (큰 사진은 긴 변 640으로 줄여 검출 — YuNet은 300픽셀 이하 얼굴에 맞게 학습됨)
#     ③ 가장 큰 얼굴 1개 선택
#     ④ (정렬 켬) 두 눈을 잇는 선이 수평이 되게 회전 → 회전된 사진에서 다시 검출
#     ⑤ 상자 + 여백(상자 크기 대비, 왼·위·오른·아래)으로 자르고 224×224로 맞춤 (학습 검증 전처리도 224×224로 늘림)
#
#   [모델 파일] models/face_detection_yunet_2026may.onnx — opencv_zoo, MIT 라이선스
#     OpenCV 5.x는 고정 크기 2023mar 파일 대신 동적 크기 2026may 파일을 써야 함 (opencv_zoo README)
#     OpenCV가 한글 경로를 못 읽어서 파일을 바이트로 읽어 넘긴다
#
#   [점검] 저장소 최상위에서:  .venv\Scripts\python 작은CNN\face_align.py
#
# #############################################################################

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2026may.onnx"
PAD_RATIO = 0.25          # 테두리 두께 = 사진 긴 변 × 0.25
DET_MAX_SIDE = 640        # 검출용으로 줄일 최대 크기
SCORE_TH = 0.6            # 얼굴 점수 기준 (기본 0.9는 꽉 찬 얼굴을 자주 놓침)

# 여백 (왼, 위, 오른, 아래) — 상자 크기 대비. face_crop_measure.py로 학습 사진(FairFace·UTKFace)에서 잰 값,
# 두 출처의 중앙값을 학습 데이터 비율 70:19로 가중 평균 (개발일지 2026-11-28_Gradio시연.md)
MARGINS = {
    # 테두리를 가장자리 복제로 바꾼 뒤 다시 잰 값 (검은 테두리일 때는 0.106, -0.104, 0.110, 0.038)
    False: (0.123, -0.141, 0.128, 0.030),   # 가중 IoU 중앙값 0.809 (FairFace 0.794, UTKFace 0.868)
    True: (0.125, -0.142, 0.126, 0.029),    # 가중 IoU 중앙값 0.795 → 원본 재현이 조금 나빠 기본값은 끔
}
DEFAULT_ALIGN = False


# =============================================================================
#   ▶ [모듈 1] 검출기 · 검출
#   - 검출기는 처음 쓸 때 한 번만 만든다. 입력 크기는 사진마다 setInputSize로 바꿈 (동적 크기 모델)
#   - 반환: (N, 15) = 상자 x,y,w,h + 눈·코·입 5점(오른눈, 왼눈, 코, 오른입꼬리, 왼입꼬리) + 점수
# =============================================================================

_det = None


def _detector():
    global _det
    if _det is None:
        buf = np.frombuffer(MODEL_PATH.read_bytes(), np.uint8)
        _det = cv2.FaceDetectorYN.create("onnx", buf, np.empty(0, np.uint8), (320, 320), SCORE_TH, 0.3, 5000)
    return _det


def detect(rgb):
    """rgb (H,W,3) uint8 → (N,15) float, 원래 좌표. 얼굴 없으면 (0,15)"""
    H, W = rgb.shape[:2]
    s = min(1.0, DET_MAX_SIDE / max(H, W))
    small = cv2.resize(rgb, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    det = _detector()
    det.setInputSize((small.shape[1], small.shape[0]))
    _, faces = det.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    if faces is None:
        return np.zeros((0, 15), np.float32)
    faces = faces.copy()
    faces[:, :14] /= s
    return faces


# =============================================================================
#   ▶ [모듈 2] 얼굴 위치 찾기 (테두리 → 검출 → 가장 큰 얼굴 → 정렬)
#   - 결과 좌표는 모두 "캔버스"(테두리 붙이고, 정렬 켬이면 회전까지 한 사진) 기준
#   - 정렬: 오른눈→왼눈 선의 기울기만큼 눈 가운데를 중심으로 회전. 회전 후 다시 검출해 상자를 새로 받고,
#     못 찾으면 원래 상자 중심만 회전 행렬로 옮겨 쓴다
#   - M: 원래 사진(테두리 포함) → 캔버스 변환 (2×3). 측정에서 원래 사진 테두리 위치를 계산할 때 씀
#   - lm: 5점 (5,2) 캔버스 좌표. 상자와 같은 검출에서 가져옴(정렬 켬이면 다시 검출한 얼굴, 못 찾으면 M으로 옮김)
#     → 히트맵 설명(영역 나누기)용으로만 쓰고 모델 입력에는 넣지 않는다
# =============================================================================

def locate(rgb, align):
    """→ dict(canvas, box(x,y,w,h), lm(5,2), n_faces, pad, M, angle) 또는 얼굴 없으면 None"""
    H, W = rgb.shape[:2]
    p = int(max(H, W) * PAD_RATIO)
    canvas = cv2.copyMakeBorder(rgb, p, p, p, p, cv2.BORDER_REPLICATE)
    faces = detect(canvas)
    if len(faces) == 0:
        return None
    f = faces[np.argmax(faces[:, 2] * faces[:, 3])]
    box, lm, M, angle = f[:4].copy(), f[4:14].reshape(5, 2).copy(), np.float32([[1, 0, 0], [0, 1, 0]]), 0.0
    if align:
        (rx, ry), (lx, ly) = f[4:6], f[6:8]
        angle = math.degrees(math.atan2(ly - ry, lx - rx))
        M = cv2.getRotationMatrix2D(((rx + lx) / 2, (ry + ly) / 2), angle, 1.0)
        canvas = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        c = M @ np.array([box[0] + box[2] / 2, box[1] + box[3] / 2, 1.0])
        again = detect(canvas)
        if len(again):
            centers = again[:, :2] + again[:, 2:4] / 2
            g = again[np.argmin(((centers - c) ** 2).sum(1))]
            box, lm = g[:4].copy(), g[4:14].reshape(5, 2).copy()
        else:
            box[:2] = c - box[2:4] / 2
            lm = np.c_[lm, np.ones(5)] @ M.T
    return dict(canvas=canvas, box=box, lm=lm, n_faces=len(faces), pad=p, M=M, angle=angle)


# =============================================================================
#   ▶ [모듈 3] 자르기
#   - 상자 + 여백 → 캔버스 안으로 자르고 224×224로 맞춤. 사진 밖으로 나간 부분은 테두리(가장자리 픽셀 복제)로
#     채워져 있음. 캔버스(사진 + 긴 변 × 0.25)보다 더 나가면 잘라냄
# =============================================================================

def crop_rect(box, margins):
    x, y, w, h = box
    ml, mt, mr, mb = margins
    return x - w * ml, y - h * mt, x + w * (1 + mr), y + h * (1 + mb)


def crop_face(image, align=DEFAULT_ALIGN, margins=None, return_landmarks=False):
    """PIL 사진 → (224×224 PIL 얼굴, 찾은 얼굴 수). 얼굴이 없으면 (None, 0)
    return_landmarks=True면 (얼굴, 수, 5점 (5,2) 224 좌표) — 자르기와 같은 변환(평행이동 + 224로 늘림)을 적용"""
    loc = locate(np.asarray(image.convert("RGB")), align)
    if loc is None:
        return (None, 0, None) if return_landmarks else (None, 0)
    cv_ = loc["canvas"]
    l, t, r, b = crop_rect(loc["box"], margins or MARGINS[align])
    l, t = max(int(round(l)), 0), max(int(round(t)), 0)
    r, b = min(int(round(r)), cv_.shape[1]), min(int(round(b)), cv_.shape[0])
    face = Image.fromarray(cv_[t:b, l:r]).resize((224, 224), Image.BILINEAR)
    if not return_landmarks:
        return face, loc["n_faces"]
    # PIL resize는 픽셀 가장자리 기준(0~W → 0~224)이라 좌표도 같은 비율로 늘리면 됨
    lm = (loc["lm"] - (l, t)) * (224 / (r - l), 224 / (b - t))
    return face, loc["n_faces"], lm


# =============================================================================
#   점검 — ① 회전한 얼굴을 정렬하면 다시 검출한 두 눈이 수평 ② 얼굴 없는 사진은 None ③ 출력 크기
# =============================================================================
if __name__ == "__main__":
    import sys
    src = Image.open(sys.argv[1]) if len(sys.argv) > 1 else None
    if src is None:
        root = Path(r"C:\3th_project\whats_your_face_age\division_224AFAD-X\validation\20s")
        src = Image.open(next(root.glob("*.chip.jpg")))
    rgb = np.asarray(src.convert("RGB"))
    tilted = np.asarray(Image.fromarray(rgb).rotate(20, expand=True))
    loc = locate(tilted, align=True)
    f = detect(loc["canvas"])
    f = f[np.argmax(f[:, 2] * f[:, 3])]
    tilt_after = math.degrees(math.atan2(f[7] - f[5], f[6] - f[4]))
    # YuNet 눈 좌표는 224 사진에서 ±6도쯤 흔들림 (같은 사진을 다시 검출해도 그 정도 차이) → 완전 0은 기대하지 않음
    assert abs(loc["angle"]) > 10 and abs(tilt_after) < 8 and abs(tilt_after) < abs(loc["angle"]) / 3, (loc["angle"], tilt_after)
    assert crop_face(Image.new("RGB", (300, 300), (128, 128, 128)))[0] is None
    face, n = crop_face(src, align=True)
    assert face.size == (224, 224) and n >= 1
    print(f"20도 기울인 사진: 검출 기울기 {loc['angle']:.1f}도 → 정렬 후 {tilt_after:.1f}도, 빈 사진 None, 점검 통과")
