# 얼굴 나이대 분류 시연 화면 (Gradio + Grad-CAM)

얼굴 사진을 올리거나 웹캠으로 찍으면 **7개 나이대 중 상위 3개와 확률**, **모델에 들어간 얼굴**, **모델이 어디를 보고 판단했는지(Grad-CAM)** 를 보여 주는 시연 화면입니다.

- **담당**: 임성빈 (LIM)
- **모델**: 작은 CNN(기본), ResNet-50, ConvNeXt-Tiny — 가중치 파일이 있는 모델만 화면의 드롭다운에 나옵니다.
- **개인정보**: 올린 사진은 디스크에 저장하지 않습니다. Gradio 내부 임시 파일은 10분마다 지워지고, 서버를 끌 때도 지워집니다. 이 PC(127.0.0.1)에서만 접속할 수 있고 외부 공유 링크는 만들지 않습니다.

## 1. 폴더 구성

```
시연/
├── README.md          ← 이 문서
├── app.py             ← 시연 화면 (모듈 G-1 ~ G-5)
├── gradcam.py         ← Grad-CAM 모듈 (C-1 ~ C-3, 다른 분석 스크립트에서 import 가능)
├── requirements.txt   ← 추가로 필요한 패키지
├── .gitignore         ← 가중치(.pth, .pt)·사진은 커밋하지 않음
└── weights/           ← 가중치 파일을 넣는 곳 (커밋 안 됨, 직접 복사)
    ├── small_cnn_best.pth
    ├── resnet50_best.pt        (있으면 드롭다운에 나옴)
    └── convnext_tiny_best.pt   (있으면 드롭다운에 나옴)
```

## 2. 실행 방법

### 2-1. 설치

작은CNN 실행 환경(torch, torchvision 등)이 있는 가상환경에서, 저장소 최상위 폴더에서:

```
pip install -r 시연\requirements.txt
```

> OpenCV는 **4.x**여야 합니다. 5.x에는 얼굴 검출에 쓰는 Haar cascade가 빠져 있습니다.

### 2-2. 가중치 넣기

`시연\weights\` 폴더를 만들고 가중치를 아래 이름으로 복사합니다.

| 모델 | 원래 파일 | 복사할 이름 |
|---|---|---|
| 작은 CNN | `작은CNN\results\<실험 이름>\small_cnn_best.pth` | `small_cnn_best.pth` |
| ResNet-50 | `ResNet50` step4_train.py가 만든 `results\<실험 이름>\best.pt` | `resnet50_best.pt` |
| ConvNeXt-Tiny | `ConvNeXt` step4_train.py가 만든 `results\<실험 이름>\best.pt` | `convnext_tiny_best.pt` |

### 2-3. 실행 (cmd)

```bat
cd /d 저장소_폴더
가상환경\Scripts\activate.bat
chcp 65001
set PYTHONIOENCODING=utf-8
python 시연\app.py
```

- `chcp 65001`, `set PYTHONIOENCODING=utf-8`: cmd에서 한글 출력이 깨지지 않게 함
- 터미널에 `Running on local URL: http://127.0.0.1:7860` 이 나오면 브라우저에서 **http://127.0.0.1:7860** 을 엽니다.
- 끄기: cmd 창에서 `Ctrl + C`

## 3. 명령줄 옵션

| 옵션 | 기본값 | 뜻 |
|---|---|---|
| `--model` | `small_cnn` | 처음 선택될 모델 (`small_cnn`, `resnet50`, `convnext_tiny`) |
| `--weights` | `시연\weights\<모델>_best.pth(.pt)` | `--model` 의 가중치 경로 (weights 폴더에 복사하지 않고 바로 쓸 때) |
| `--device` | GPU가 있으면 `cuda`, 없으면 `cpu` | 계산 장치 |
| `--port` | `7860` | 브라우저 주소의 포트 번호 |
| `--margin` | `0.15` | 얼굴 상자 여백 (상자 크기 대비 비율). 학습 사진과 비슷한 비율이 되도록 정한 값 |
| `--image` | 없음 | 사진 경로를 주면 화면 없이 상위 3개만 출력 (점검용) |

예시:

```
python 시연\app.py --device cpu --port 7861
python 시연\app.py --model small_cnn --weights 작은CNN\results\small_cnn_main\small_cnn_best.pth
python 시연\app.py --image 사진.jpg
python 시연\gradcam.py --image 사진.jpg --out 결과.png      ← 히트맵 한 장 저장 (얼굴 | 겹친 이미지)
```

## 4. 가중치 바꾸기 (재학습 후)

**코드는 고치지 않습니다.** 새 가중치 파일을 `시연\weights\` 의 같은 이름으로 덮어쓰고 다시 실행하면 됩니다.
모델 구조(SmallCNN은 `작은CNN\CNN_4layers_unified.py`에서 import, 팀원 모델은 `timm.create_model`)가 바뀌지 않았다면 그대로 불러옵니다.

## 5. 처리 순서

1. **얼굴 검출·자르기** (`detect_and_crop`) — OpenCV Haar cascade (임시)
   - 사진 둘레에 테두리를 붙이고 검출 → 얼굴로 꽉 찬 사진도 찾음
   - 정면 → 정면(alt2) → 옆얼굴 → 옆얼굴(좌우 반전) 순서로 시도, 여러 얼굴이면 가장 큰 얼굴
   - 못 찾으면 "얼굴을 찾지 못했습니다. 정면 사진으로 다시 시도해 주세요"
   - 테스트 사진 500장 중 448장(90%) 검출. 선글라스, 많이 돌아간 얼굴, 아주 어두운 사진은 못 찾음
2. **전처리** — 학습 코드의 검증용 transform 그대로 (224×224 → 텐서 → ImageNet 평균·표준편차 정규화)
3. **예측** — softmax 확률 7개 → 상위 3개 표시
4. **Grad-CAM** — 예측 1위 나이대의 히트맵을 얼굴 위에 겹침 (빨강 = 많이 본 곳)

| 모델 | Grad-CAM 대상 층 | 특징 맵 모양 |
|---|---|---|
| 작은 CNN | 4번째 블록 최종 출력 (`gap`의 입력) | (1, 256, 14, 14) |
| ResNet-50 | `layer4` 출력 | (1, 2048, 7, 7) |
| ConvNeXt-Tiny | `stages[-1]` 출력 | (1, 768, 7, 7) |

## 6. 남은 일

- [ ] 얼굴 검출을 팀 전처리와 같은 검출·정렬 방법으로 교체 (`app.py`의 `# TODO` 위치, 함수 하나만 바꾸면 됨)
- [ ] 새 데이터셋으로 재학습한 가중치로 교체
- [ ] ResNet-50·ConvNeXt 가중치로 드롭다운·Grad-CAM 실제 확인
