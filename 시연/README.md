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

**작은 CNN (기본: v2 가중치)** — `작은CNN\results\small_cnn_v2_lr1e-3\small_cnn_best.pth` 를 바로 읽습니다.
새 데이터(`splits_csv_v2`)로 학습한 17 에폭 가중치입니다 (검증 macro-F1 0.461, 작은CNN 개발일지 14).
가중치 파일은 저장소에 올리지 않으므로, 학습한 PC의 `results\small_cnn_v2_lr1e-3\small_cnn_best.pth` 를 위 위치에 복사합니다.

```
mkdir 작은CNN\results\small_cnn_v2_lr1e-3
copy <학습 폴더>\results\small_cnn_v2_lr1e-3\small_cnn_best.pth 작은CNN\results\small_cnn_v2_lr1e-3\
```

예전 가중치나 다른 실험 가중치는 `--weights` 로 지정합니다 (예: `--weights 시연\weights\small_cnn_best.pth`).

**팀원 모델** — `시연\weights\` 폴더에 아래 이름으로 복사합니다. 파일이 없으면 드롭다운에 나오지 않습니다.

| 모델 | 원래 파일 | 복사할 이름 |
|---|---|---|
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
| `--weights` | 작은 CNN: `작은CNN\results\small_cnn_v2_lr1e-3\small_cnn_best.pth`<br>팀원 모델: `시연\weights\<모델>_best.pt` | `--model` 의 가중치 경로 |
| `--device` | GPU가 있으면 `cuda`, 없으면 `cpu` | 계산 장치 |
| `--port` | `7860` | 브라우저 주소의 포트 번호 |
| `--margin` | `0.02 -0.065 0.02 0.105` | 얼굴 상자 여백 **왼 위 오른 아래** (상자 크기 대비 비율, 음수 = 상자 안쪽으로 자름). v2 학습 사진에서 잰 값 (5-1 참고) |
| `--image` | 없음 | 사진 경로를 주면 화면 없이 상위 3개만 출력 (점검용) |

예시:

```
python 시연\app.py --device cpu --port 7861
python 시연\app.py --model small_cnn --weights 작은CNN\results\small_cnn_main\small_cnn_best.pth   ← 예전 데이터 가중치
python 시연\app.py --image 사진.jpg
python 시연\app.py --image 사진.jpg --margin 0.15 0.15 0.15 0.15                                  ← 예전 여백
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

### 5-1. 얼굴 자르는 여백 (v2 학습 사진에 맞춤)

v2 학습 사진(`division_224AFAD-X/training`) 나이대별 300장 = 2,100장에 `detect_face`를 돌려 Haar 상자를 쟀습니다 (검출 1,795장, 중앙값).

| 측정값 | 값 |
|---|---|
| 상자 폭 ÷ 사진 폭 | **0.96** (예전 데이터는 약 0.79) |
| 상자 중심 (x, y) | 0.49, **0.42** (상자가 위쪽에 치우침) |
| 상자 밖 여백 (상자 크기 대비) | 왼 0.014, 위 **−0.065**, 오른 0.028, 아래 **0.105** (나이대별 차이 ±0.01) |

→ 여백을 **사방 0.15 → (왼 0.02, 위 −0.065, 오른 0.02, 아래 0.105)** 로 바꿨습니다. 학습 사진은 얼굴 상자와 거의 같은 크기로, 위는 상자 안쪽에서·아래는 턱 아래까지 잘려 있습니다.

**점검** — 검증 사진을 원본 224 그대로 넣었을 때와, 아래처럼 바꾼 뒤 `detect_and_crop` → 예측했을 때 1위 나이대가 같은 장수 (작은 CNN v2, CPU):

| 시험 | 장수 | 검출 | 새 여백 | 예전 0.15 |
|---|---:|---:|---:|---:|
| 2배 확대 (448×448) | 10 | 9 | 7 (검출 중 78%) | 8 (89%) |
| 2배 확대 (448×448) | 210 (나이대별 30) | 180 | 145 (81%) | 153 (85%) |
| 448 회색 바탕 가운데에 원본 (얼굴 주변에 배경이 있는 사진) | 10 | 7 | **5 (71%)** | 2 (29%) |
| 448 회색 바탕 가운데에 원본 | 210 | 177 | **132 (75%)** | 106 (60%) |

- 2배 확대만 하면 예전 여백이 조금 더 같게 나옵니다. 학습 사진은 얼굴이 사진을 거의 채워서, 여백 0.15로 넓혀도 사진 끝에서 잘려 결국 **원본 전체**가 들어가기 때문입니다.
- 실제 시연처럼 얼굴 주변에 배경이 있으면 새 여백이 원본과 훨씬 비슷하게 예측합니다 (검출 중 75% vs 60%).
- 새 여백으로도 검출된 사진 4장 중 1장은 1위 나이대가 달라집니다. 예를 들어 `30s/62771.jpg`는 30대 0.202, 40대 0.195처럼 1·2위 확률 차이가 작아, 이런 사진은 자르는 위치가 조금만 달라도 1위가 바뀔 수 있습니다.

## 6. 남은 일

- [ ] 얼굴 검출을 팀 전처리와 같은 검출·정렬 방법으로 교체 (`app.py`의 `# TODO` 위치, 함수 하나만 바꾸면 됨)
- [x] 새 데이터셋(v2)으로 재학습한 작은 CNN 가중치로 교체
- [ ] ResNet-50·ConvNeXt v2 가중치로 드롭다운·Grad-CAM 실제 확인 (파일이 생기면 `시연\weights\` 에 복사)
