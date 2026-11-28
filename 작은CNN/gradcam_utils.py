# #############################################################################
#
#   Grad-CAM 공용 함수 — 작은 CNN · ConvNeXt-Tiny · ResNet-50 세 모델 공통
#   인공지능 4조 · 담당: LIM
#
#   [왜 새 파일인가]
#     시연/gradcam.py는 사진 1장씩만 계산한다. 세 모델 × 수천 장을 비교하려면 배치로 한 번에 계산해야 해서
#     같은 알고리즘을 배치 버전으로 다시 정리했다. Gradio에서도 이 파일 하나만 import하면 되게
#     모델 읽기(load_model) · 전처리(preprocess) · 히트맵 겹치기(overlay)까지 함께 넣었다.
#
#   [Grad-CAM 계산 순서]
#     ① 대상 층의 특징 맵 A (B, C, h, w)를 forward hook으로 저장
#     ② 고른 나이대 점수를 역전파 → A에 대한 기울기 dScore/dA를 텐서 hook으로 저장
#     ③ 채널별 중요도 = 기울기의 공간 평균 (채널마다 숫자 1개)
#     ④ 히트맵 = ReLU(Σ 중요도 × A) → 224×224로 키움 → 사진마다 최댓값으로 나눠 0~1
#
#   [다른 스크립트·Gradio에서 쓰는 법]
#     from gradcam_utils import load_model, GradCAM, preprocess, overlay, BANDS
#     model = load_model("convnext_tiny", "weights/convnext_tiny/best.pt")
#     cam = GradCAM(model, "convnext_tiny")
#     heat, pred, prob = cam(preprocess([pil_face]))     # heat: (1, 224, 224) 0~1
#     img = overlay(pil_face, heat[0])                    # PIL 이미지
#
#   [점검] 저장소 최상위에서:  .venv\Scripts\python 작은CNN\gradcam_utils.py
#
# #############################################################################

import os
import sys

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)                                     # CNN_4layers_unified.py (작은 CNN 구조)
sys.path.insert(0, os.path.join(ROOT, "ConvNeXt", "code"))  # 팀원 common.py (나이대 순서, 검증 전처리)
from common import BANDS, build_transforms                   # noqa: E402  BANDS: 0=0-9세 … 6=60세 이상
from CNN_4layers_unified import SmallCNN                     # noqa: E402

MODEL_NAMES = ["small_cnn", "convnext_tiny", "resnet50"]
DEFAULT_WEIGHTS = {
    "small_cnn": os.path.join(HERE, "results", "small_cnn_v2_lr1e-3", "best.pt"),
    "convnext_tiny": os.path.join(ROOT, "weights", "convnext_tiny", "best.pt"),
    "resnet50": os.path.join(ROOT, "weights", "resnet50", "best.pt"),
}
VAL_TF = build_transforms(train=False)   # 팀 공통 검증 전처리: Resize 224 → ToTensor → ImageNet 정규화


# =============================================================================
#   ▶ [모듈 1] 모델 읽기 · 전처리
#   - 팀원 best.pt와 내 best.pt 모두 {"model": state_dict, ...} 형식. 옛 .pth(가중치만)도 읽히게 둘 다 처리
#   - strict 로드: 층 이름이 하나라도 다르면 바로 에러 → 엉뚱한 가중치로 히트맵을 그리는 일 방지
#   - eval(): Dropout 끄고 BatchNorm을 저장된 통계로 고정. 그래야 배치 안 사진끼리 서로 영향이 없음
# =============================================================================

def load_model(name, weights=None, device="cpu"):
    model = SmallCNN() if name == "small_cnn" else timm.create_model(name, pretrained=False, num_classes=len(BANDS))
    ckpt = torch.load(weights or DEFAULT_WEIGHTS[name], map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
    return model.to(device).eval()


def preprocess(pil_images):
    """PIL 사진 리스트 → (B, 3, 224, 224) 정규화 텐서 (학습 때 검증 전처리와 똑같이)"""
    return torch.stack([VAL_TF(im.convert("RGB")) for im in pil_images])


# =============================================================================
#   ▶ [모듈 2] 대상 층 고르기 — 각 모델의 "마지막 합성곱 블록 출력"
#   - 마지막 블록을 쓰는 이유: 앞쪽 층은 선·모서리 같은 단순 무늬, 마지막 블록은 나이대 판단에 직접 쓰이는
#     고수준 특징이면서 아직 위치 정보(가로×세로 격자)가 남아 있는 마지막 지점이라서
#   - small_cnn : gap의 "입력" (256×14×14) = 4번째 블록 Conv→BN→ReLU→MaxPool의 최종 출력
#                 self.pool은 4개 블록이 같이 쓰는 모듈이라 거기에 hook을 걸면 4번 불림 → 한 번만 불리는 gap 입력을 봄
#   - convnext  : stages[-1] 출력 (768×7×7) = 마지막 stage(ConvNeXt 블록 3개) 출력, 그다음은 풀링·head뿐
#   - resnet50  : layer4 출력 (2048×7×7) = 마지막 Bottleneck 블록의 잔차 더하기 + ReLU 이후
#   - 팀원 common.py의 gradcam_target_layer (layer4[-1], stages[-1])와 같은 위치
# =============================================================================

def target_layer(model, name):
    if name == "small_cnn":
        return model.gap, True              # (층, 층의 입력을 볼지)
    if name.startswith("convnext"):
        return model.stages[-1], False
    if name.startswith("resnet"):
        return model.layer4, False
    raise ValueError(f"Grad-CAM 대상 층이 정해지지 않은 모델: {name}")


# =============================================================================
#   ▶ [모듈 3] 배치 Grad-CAM
#   - 기울기는 module backward hook 대신 "텐서 hook"으로 받는다. ResNet 안의 ReLU(inplace=True) 때문에
#     module backward hook은 경고·오류가 날 수 있는데, 텐서 hook은 그 영향을 받지 않는다
#   - 배치 역전파: 사진마다 고른 나이대 점수를 더해서 한 번에 backward. eval 모드에서는 사진끼리 계산이
#     섞이지 않으므로, 합의 기울기 = 사진별 기울기 (점검에서 1장씩 계산한 값과 같은지 확인)
#   - 정규화: 사진마다 최댓값으로만 나눔(최솟값은 빼지 않음). ReLU 뒤라 0이 "안 본 곳"이라는 뜻을 유지해야
#     영역별 에너지 비율(합÷합)이 원래 히트맵과 같게 나온다
# =============================================================================

class GradCAM:
    def __init__(self, model, name):
        self.model = model
        layer, self.use_input = target_layer(model, name)
        self.act = self.grad = None
        self.hook = layer.register_forward_hook(self._save)

    def _save(self, module, inp, out):
        a = inp[0] if self.use_input else out
        self.act = a.detach()
        a.register_hook(lambda g: setattr(self, "grad", g.detach()))   # 역전파 때 이 텐서의 기울기 저장

    def __call__(self, x, class_idx=None):
        """x: (B,3,224,224) → (히트맵 (B,224,224) 0~1 numpy, 사용한 나이대 (B,), 예측 확률 (B,7))
        class_idx=None이면 모델의 예측 1위 나이대 기준"""
        x = x.to(next(self.model.parameters()).device)
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():                          # Gradio 등에서 no_grad 안에서 불러도 기울기 계산되게
            logits = self.model(x)
            cls = logits.argmax(1) if class_idx is None else torch.as_tensor(class_idx, device=x.device).view(-1)
            logits.gather(1, cls[:, None]).sum().backward()
        w = self.grad.mean(dim=(2, 3), keepdim=True)                       # ③ (B, C, 1, 1)
        cam = F.relu((w * self.act).sum(1, keepdim=True))                  # ④ (B, 1, h, w)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[:, 0]
        cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)    # 전부 0이면 0으로 남음
        return cam.cpu().numpy(), cls.cpu().numpy(), logits.softmax(1).detach().cpu().numpy()

    def remove(self):
        self.hook.remove()


# =============================================================================
#   ▶ [모듈 4] 히트맵 겹치기 · 영역 에너지
#   - overlay: 빨강 = 많이 본 곳, 파랑 = 적게 본 곳 (시연/gradcam.py와 같은 색)
#   - center_ratio: 가운데 정사각형(가로·세로 각 frac, 기본 60% → 면적 35.8%) 안 히트맵 합 ÷ 전체 합
#     사진마다 고르게 퍼진 히트맵이면 면적 비율(35.8%)과 같게 나오므로, 이 값과 비교해서 읽는다
# =============================================================================

def overlay(face, cam, alpha=0.4):
    base = np.asarray(face.convert("RGB").resize((224, 224)), dtype=np.float32)
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return Image.fromarray(np.uint8((1 - alpha) * base + alpha * heat))


def center_mask(frac=0.6, size=224):
    m = np.zeros((size, size), bool)
    lo = round(size * (1 - frac) / 2)
    m[lo:size - lo, lo:size - lo] = True
    return m


def center_ratio(cams, frac=0.6):
    """cams: (B,224,224) → (B,) 가운데 에너지 비율. 히트맵이 전부 0이면 NaN"""
    m = center_mask(frac, cams.shape[-1])
    total = cams.sum(axis=(1, 2))
    return np.where(total > 0, cams[:, m].sum(1) / np.maximum(total, 1e-12), np.nan)


# =============================================================================
#   점검 — 세 모델 모두 ① 대상 특징 맵 크기 ② 배치 결과 = 1장씩 결과 ③ 값 범위 ④ 영역 비율 계산
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(3, 3, 224, 224)
    expect = {"small_cnn": (256, 14, 14), "convnext_tiny": (768, 7, 7), "resnet50": (2048, 7, 7)}
    for name in MODEL_NAMES:
        cam = GradCAM(load_model(name), name)
        heat, cls, prob = cam(x)
        assert tuple(cam.act.shape[1:]) == expect[name], (name, cam.act.shape)
        assert heat.shape == (3, 224, 224) and heat.min() >= 0 and heat.max() <= 1
        one, cls1, _ = cam(x[1:2])
        assert cls1[0] == cls[1] and np.allclose(one[0], heat[1], atol=1e-4), f"{name}: 배치와 1장 결과가 다름"
        print(f"{name}: 특징 맵 {tuple(cam.act.shape[1:])}, 배치=1장 결과 일치, 예측 {cls.tolist()}")
    flat = np.ones((1, 224, 224))
    assert abs(center_ratio(flat)[0] - center_mask().mean()) < 1e-9 and np.isnan(center_ratio(np.zeros((1, 224, 224)))[0])
    print(f"가운데 60% 상자 면적 비율 {center_mask().mean():.3f}, 점검 통과")
