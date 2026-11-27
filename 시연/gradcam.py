# #############################################################################
#
#   Grad-CAM — 모델이 사진의 어디를 보고 나이대를 골랐는지 히트맵으로 보기
#   인공지능 4조 · 담당: LIM
#   외부 Grad-CAM 라이브러리 없이 forward hook + backward hook으로 직접 구현
#
#   [Grad-CAM 계산 순서]
#     ① 대상 층의 출력(특징 맵 A, 채널 C개)을 forward hook으로 저장
#     ② 고른 나이대 점수 y를 backward → 대상 층 출력에 대한 기울기 dy/dA를 backward hook으로 저장
#     ③ 채널별 중요도 = 기울기의 공간 평균 (채널마다 숫자 1개)
#     ④ 히트맵 = ReLU(Σ 중요도 × A) — 점수를 "올리는" 쪽만 남김
#     ⑤ 0~1 정규화 → 224×224로 키우기 → 얼굴 위에 컬러맵으로 겹치기
#
#   [모듈 구성]
#     C-1 대상 층 고르기   C-2 GradCAM 클래스   C-3 얼굴 위에 겹치기
#
#   [다른 스크립트에서 쓰는 법]
#     from gradcam import GradCAM, target_layer, overlay
#     cam = GradCAM(model, *target_layer(model, "small_cnn"))
#     heat, cls = cam(x)                 # x: (1, 3, 224, 224) 정규화된 텐서
#     img = overlay(face, heat)          # face: PIL 얼굴 → PIL 겹친 이미지
#
#   [점검] 저장소 최상위 폴더에서
#     python 시연\gradcam.py --image 사진.jpg --out 결과.png
#
# #############################################################################


# =============================================================================
#   ▶ [모듈 C-1 시작] 대상 층 고르기
#   입력: 모델, 모델 이름
#   출력: (hook을 걸 층, 층의 입력을 볼지 여부)
# =============================================================================

import cv2                                              # 컬러맵 입히기
import numpy as np                                      # 배열 계산
import torch                                            # PyTorch 본체
import torch.nn.functional as F                         # interpolate(크기 키우기), relu
from PIL import Image                                   # 결과 이미지


def target_layer(model, name):                          # (model=모델, name=모델 이름) → (층, use_input)
    if name == "small_cnn":
        # SmallCNN의 self.pool은 4개 블록이 같이 쓰는 모듈이라 hook을 걸면 4번 불림.
        # 그래서 한 번만 불리는 gap의 "입력" = 4번째 블록 최종 출력(ReLU → MaxPool 후)을 봄
        return model.gap, True                          # gap 입력: (1, 256, 14, 14)
    if name == "resnet50":
        return model.layer4, False                      # layer4 출력: (1, 2048, 7, 7)
    if name == "convnext_tiny":
        return model.stages[-1], False                  # 마지막 stage 출력: (1, 768, 7, 7)
    raise ValueError(f"Grad-CAM 대상 층이 정해지지 않은 모델: {name}")

# -----------------------------------------------------------------------------
# ◀ [모듈 C-1 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   ▶ [모듈 C-2 시작] GradCAM 클래스
#   입력: (1, 3, 224, 224) 정규화된 텐서, 나이대 번호(없으면 예측 1위)
#   출력: (224, 224) 0~1 히트맵(numpy), 사용한 나이대 번호
# =============================================================================

class GradCAM:
    def __init__(self, model, layer, use_input=False):  # (model=eval() 상태 모델, layer=대상 층, use_input=층의 입력을 볼지)
        self.model = model                              # 모델 저장
        self.use_input = use_input                      # True면 층 입력, False면 층 출력
        self.act = None                                 # ① 특징 맵 A 저장 자리
        self.grad = None                                # ② 기울기 dy/dA 저장 자리
        # forward hook: 층을 지날 때마다 (모듈, 입력 튜플, 출력)을 받아 호출됨
        self.h1 = layer.register_forward_hook(self._save_act)
        # backward hook: 역전파가 층을 지날 때 (모듈, 입력 쪽 기울기 튜플, 출력 쪽 기울기 튜플)을 받아 호출됨
        self.h2 = layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):              # (module=층, inp=입력 튜플, out=출력)
        self.act = (inp[0] if self.use_input else out).detach()   # (1, C, h, w) 저장 (detach: 계산 기록에서 떼어 냄)

    def _save_grad(self, module, grad_in, grad_out):    # (module=층, grad_in=입력 쪽 기울기, grad_out=출력 쪽 기울기)
        self.grad = (grad_in[0] if self.use_input else grad_out[0]).detach()   # (1, C, h, w) 저장

    def __call__(self, x, class_idx=None):              # (x=(1, 3, 224, 224) 텐서, class_idx=나이대 번호, None이면 1위)
        # torch.no_grad()를 쓰면 기울기를 못 구하므로 쓰지 않음. 모델은 eval() 상태여야 함 (Dropout 끔, BN 고정)
        self.model.zero_grad(set_to_none=True)          # 이전 기울기 지우기
        logits = self.model(x)                          # (1, 7) 점수 — 이때 ① act 저장됨
        if class_idx is None:                           # 나이대를 안 정했으면
            class_idx = int(logits.argmax(dim=1))       # 예측 1위
        logits[0, class_idx].backward()                 # 고른 나이대 점수 하나만 역전파 — 이때 ② grad 저장됨

        weights = self.grad.mean(dim=(2, 3), keepdim=True)          # ③ (1, C, 1, 1) 채널별 중요도
        cam = F.relu((weights * self.act).sum(dim=1, keepdim=True)) # ④ (1, 1, h, w) 가중합 → 음수 버림
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)   # (1, 1, 224, 224) 크기 키우기
        cam = cam[0, 0]                                             # (224, 224)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)    # ⑤ 0~1 정규화 (1e-8: 전부 0일 때 0 나눗셈 방지)
        return cam.cpu().numpy(), class_idx                         # (히트맵, 나이대 번호)

    def remove(self):                                   # hook 떼기 (모델을 다른 용도로 계속 쓸 때)
        self.h1.remove()
        self.h2.remove()

# -----------------------------------------------------------------------------
# ◀ [모듈 C-2 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   ▶ [모듈 C-3 시작] 얼굴 위에 겹치기
#   입력: PIL 얼굴, (224, 224) 0~1 히트맵
#   출력: 224×224 PIL 이미지 (빨강 = 많이 본 곳, 파랑 = 적게 본 곳)
# =============================================================================

def overlay(face, cam, alpha=0.4):                      # (face=PIL 얼굴, cam=0~1 히트맵, alpha=히트맵 진하기 0~1)
    base = np.array(face.convert("RGB").resize((224, 224)), dtype=np.float32)   # (224, 224, 3) 모델이 본 크기와 같게
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)             # (224, 224, 3) 0~255 → 파랑~빨강, BGR 순서
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32)             # BGR → RGB
    mixed = (1 - alpha) * base + alpha * heat           # 두 이미지를 섞기
    return Image.fromarray(np.uint8(mixed))             # 배열 → PIL

# -----------------------------------------------------------------------------
# ◀ [모듈 C-3 끝]
# -----------------------------------------------------------------------------


# =============================================================================
#   점검 — 사진 한 장으로 히트맵을 만들어 파일로 저장
# =============================================================================
if __name__ == "__main__":
    import argparse                                     # 명령줄 옵션
    from app import BAND_NAMES, DEFAULT_WEIGHTS, detect_and_crop, load_model, val_transform   # 시연 화면과 같은 함수 사용

    p = argparse.ArgumentParser(description="Grad-CAM 한 장 점검")
    p.add_argument("--image", required=True)            # 얼굴 사진 경로
    p.add_argument("--out", default="gradcam_check.png")   # 저장할 파일 (원본 얼굴 | 겹친 이미지)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")   # 장치
    args = p.parse_args()

    model = load_model("small_cnn", DEFAULT_WEIGHTS["small_cnn"], args.device)   # 모델 (eval 상태)
    face = detect_and_crop(Image.open(args.image))      # 얼굴 잘라내기
    assert face is not None, "얼굴을 찾지 못했습니다"
    x = val_transform(face).unsqueeze(0).to(args.device)    # (1, 3, 224, 224)
    cam = GradCAM(model, *target_layer(model, "small_cnn")) # hook 걸기
    heat, cls = cam(x)                                  # 히트맵, 예측 1위 번호
    assert cam.act.shape == (1, 256, 14, 14), cam.act.shape   # 대상 특징 맵 모양 확인
    assert heat.shape == (224, 224) and heat.min() >= 0 and heat.max() <= 1   # 크기·범위 확인

    both = Image.new("RGB", (448, 224))                 # 왼쪽 얼굴, 오른쪽 겹친 이미지
    both.paste(face.resize((224, 224)), (0, 0))
    both.paste(overlay(face, heat), (224, 0))
    both.save(args.out)                                 # 저장
    print(f"예측 1위 {BAND_NAMES[cls]} → {args.out} 저장")

# #############################################################################
# 모듈 끝: gradcam.py
# #############################################################################
