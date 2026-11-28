# 얼굴 나이대 분류 시연 화면 (Gradio + YuNet + Grad-CAM)

얼굴 사진을 올리거나 웹캠으로 찍으면 아래를 보여 주는 시연 화면입니다.

- **7개 나이대 중 상위 3개와 확률**
- **잘라낸 얼굴**: 모델에 들어가는 224×224 사진 그대로
- **Grad-CAM**: 모델이 어디를 보고 판단했는지 (빨강 = 많이 본 곳, 예측 1위 나이대 기준)
- **세 모델 비교**: 작은 CNN · ConvNeXt-Tiny · ResNet-50을 한 줄에 나란히
- **글 설명**: 얼굴 영역별 주목 비중 (예: "오른쪽 볼 29% · 기타(머리·배경) 28% · …")
- **가림 실험** (켤 때만): 영역마다 그 부분만 블러한 사진으로 다시 예측해 확률이 얼마나 떨어지는지 표로

**담당**: 임성빈 (LIM)

> 실제 시연 앱은 **`작은CNN/app_gradio.py`** 입니다. 이 폴더(`시연/`)의 `app.py`는 예전 Haar 방식 앱이며 맨 아래 [10. 예전 버전](#10-예전-버전-시연apppy)에 짧게 남겨 두었습니다.
> 근거 문서: [개발일지 Gradio 시연](../개발일지/LIM/2026-11-28_Gradio시연.md), [개발일지 20 Grad-CAM 글 설명](../개발일지/LIM/2026-11-28_20_GradCAM_글설명.md)

## 1. 파일 구성

**지금 쓰는 앱 (`작은CNN/`)**

| 파일 | 역할 |
|---|---|
| [`작은CNN/app_gradio.py`](../작은CNN/app_gradio.py) | 시연 화면. `--selftest`로 점검 |
| [`작은CNN/face_align.py`](../작은CNN/face_align.py) | YuNet 검출 → 가장 큰 얼굴 → (눈 수평 정렬) → 여백으로 자르기 → 224. 실행하면 정렬 점검 |
| [`작은CNN/face_regions.py`](../작은CNN/face_regions.py) | YuNet 5점으로 얼굴 영역 나누기 · 영역별 주목 비중 · 가림 실험 |
| [`작은CNN/gradcam_utils.py`](../작은CNN/gradcam_utils.py) | 모델 읽기, 팀 검증 전처리, Grad-CAM, 히트맵 겹치기 |
| [`작은CNN/face_crop_measure.py`](../작은CNN/face_crop_measure.py) | 학습 사진에서 YuNet 상자 기준 여백을 다시 재는 점검 스크립트 |
| `작은CNN/models/face_detection_yunet_2026may.onnx` | YuNet 얼굴 검출 모델 (opencv_zoo) |
| [`작은CNN/models/LICENSE_YuNet.txt`](../작은CNN/models/LICENSE_YuNet.txt) | YuNet MIT 라이선스 원문 |

**예전 앱 (`시연/`, 지금은 안 씀)**

```
시연/
├── README.md          ← 이 문서
├── app.py             ← 예전 시연 화면 (Haar 검출)
├── gradcam.py         ← 예전 Grad-CAM (사진 1장씩)
├── requirements.txt   ← 예전 앱용 패키지 (OpenCV 4.x)
├── tools/             ← 예전 여백 측정·비교 스크립트
└── weights/           ← 예전 앱용 가중치 자리 (커밋 안 됨)
```

## 2. 설치

저장소 최상위의 `requirements.txt` 하나로 설치합니다 (가상환경 `.venv`). 시연 앱만을 위한 추가 패키지는 없습니다.

```bat
cd /d 저장소_폴더
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

- 이 파일에 `opencv-python-headless 5.1.0.95`, `gradio 6.32.0`이 들어 있습니다. **새 앱은 OpenCV 5.1에서 동작합니다** (YuNet은 5.x용 동적 크기 모델 `2026may` 사용).
- `시연\requirements.txt`는 예전 앱용입니다. 새 앱에는 필요 없습니다.

## 3. 가중치 넣기

앱은 켤 때 **세 모델을 모두 읽습니다.** 하나라도 파일이 없으면 앱이 시작되지 않습니다 (`FileNotFoundError`).
경로는 `gradcam_utils.py`의 `DEFAULT_WEIGHTS` 그대로입니다 (저장소 최상위 기준).

| 모델 | 읽는 경로 | git에 있나 |
|---|---|---|
| 작은 CNN | `작은CNN\results\small_cnn_v2_lr1e-3\small_cnn_best.pth` | **있음** — clone하면 바로 읽힘 |
| ConvNeXt-Tiny | `weights\convnext_tiny\best.pt` | 없음 — 직접 복사 |
| ResNet-50 | `weights\resnet50\best.pt` | 없음 — 직접 복사 |

- **팀원 가중치**는 `.gitignore`(`weights/`, `*.pt`) 때문에 저장소에 없습니다. 팀원에게 받은 `best.pt`(팀원 실험 폴더 `results\v2_convnext\`, `results\v2_resnet50\`)를 위 이름으로 복사하세요.
- 새로 clone한 PC에서는 **ConvNeXt-Tiny · ResNet-50 가중치 두 개만 복사**하면 앱이 켜집니다.

## 4. 실행 (Windows cmd, 저장소 최상위에서)

```bat
chcp 65001
set PYTHONIOENCODING=utf-8

:: 기본 실행 → 브라우저에서 http://127.0.0.1:7860
.venv\Scripts\python 작은CNN\app_gradio.py

:: 포트 바꾸기 (7860이 이미 쓰이는 중일 때)
.venv\Scripts\python 작은CNN\app_gradio.py --port 7861

:: 점검: 전처리 일치, 합성 사진 검출, 처리 시간
.venv\Scripts\python 작은CNN\app_gradio.py --selftest --image-root C:\3th_project\whats_your_face_age\division_224AFAD-X

:: 얼굴 정렬 점검 / 여백 다시 재기
.venv\Scripts\python 작은CNN\face_align.py
.venv\Scripts\python 작은CNN\face_crop_measure.py --image-root C:\3th_project\whats_your_face_age\division_224AFAD-X
```

- `chcp 65001`, `set PYTHONIOENCODING=utf-8`: cmd에서 한글 출력이 깨지지 않게 함
- `--image-root`는 `division_224AFAD-X` 사진 폴더 위치입니다 (사진은 저장소에 없음). 위 경로는 예시입니다.
- 끄기: cmd 창에서 **Ctrl + C** (이렇게 꺼야 임시 폴더가 바로 지워집니다)

## 5. 명령줄 옵션 (`app_gradio.py`)

| 옵션 | 기본값 | 뜻 |
|---|---|---|
| `--selftest` | 끔 | 화면 없이 점검만 실행 (전처리 일치 · 합성 사진 30장 · 처리 시간) |
| `--image-root` | 없음 | `--selftest`에 쓸 `division_224AFAD-X` 폴더 |
| `--port` | `7860` | 브라우저 주소의 포트 번호 |
| `--share` | 끔 | Gradio 공유 링크(`https://xxxx.gradio.live`) 생성 — 인터넷 어디서나 접속 |
| `--lan` | 끔 | 같은 와이파이·네트워크에서 접속 허용(`0.0.0.0`). 실행 때 이 PC의 접속 주소 출력 |
| `--auth 아이디:비번` | 없음 | 로그인 화면 걸기. 형식이 틀리면 바로 멈춤 |

```bat
.venv\Scripts\python 작은CNN\app_gradio.py --lan --auth team4:비밀번호     :: 같은 네트워크 팀원
.venv\Scripts\python 작은CNN\app_gradio.py --share --auth team4:비밀번호   :: 인터넷 공유 링크
```

> **주의**
> - `--share`·`--lan`을 쓸 때는 **`--auth`를 꼭 같이** 쓰세요. 없으면 주소를 아는 누구나 접속해 사진을 올릴 수 있습니다 (앱도 경고를 출력합니다).
> - `--share`를 켜면 **사진이 Gradio 공유 서버(gradio.live)를 거쳐** 전달됩니다. 파일로 남는 곳은 여전히 이 PC의 임시 폴더뿐입니다.
> - 공유 링크 만들기에 실패하면 "[공유 링크 실패] …"를 출력하고 이 PC 전용으로 계속 동작합니다. 같은 네트워크 팀원에게는 `--lan`을 쓰세요.
> - `--lan`을 처음 쓰면 Windows 방화벽 창이 뜰 수 있습니다. "개인 네트워크"만 허용합니다.

## 6. 처리 순서

1. **얼굴 검출** (`face_align.py`) — YuNet (`cv2.FaceDetectorYN`)
   - 사진 둘레에 긴 변 × 0.25 두께의 테두리를 **가장자리 픽셀 복제**로 붙임 → 얼굴이 사진 끝까지 찬 사진도 검출
   - 큰 사진은 긴 변 640으로 줄여 검출, 얼굴 점수 기준 0.6
2. **가장 큰 얼굴 1개** 선택 (여러 개면 화면에 "얼굴 N개 중 가장 큰 얼굴")
3. **(정렬 선택)** "얼굴 정렬 (두 눈 수평)" 체크박스, 기본 끔. 켜면 두 눈이 수평이 되게 회전 후 다시 검출
4. **여백 자르기** — 학습 사진(FairFace·UTKFace)에서 거꾸로 잰 여백, 상자 크기 대비 왼 0.123 · 위 −0.141 · 오른 0.128 · 아래 0.030 (정렬 켬: 0.125 · −0.142 · 0.126 · 0.029)
5. **224×224**로 맞춤
6. **팀 검증 전처리** (`gradcam_utils.preprocess`) — Resize 224 → ToTensor → ImageNet 정규화 (팀원 `common.py`의 검증 transform 그대로)
7. **예측 + Grad-CAM** — softmax 확률 상위 3개, 히트맵은 각 모델의 **예측 1위 나이대** 기준
8. **영역별 주목 비중** (`face_regions.py`) — YuNet 눈·코·입 5점으로 얼굴을 9개 영역(이마, 왼쪽/오른쪽 눈가, 코, 입, 왼쪽/오른쪽 볼, 턱, 기타(머리·배경))으로 나누고, 영역별 히트맵 합 ÷ 전체 합. 가림 실험을 켜면 영역마다 블러(반지름 16)한 9장을 다시 예측
   - **5점은 설명용으로만 씁니다. 모델 입력에는 들어가지 않습니다.**

| 모델 | Grad-CAM 대상 층 | 특징 맵 모양 |
|---|---|---|
| 작은 CNN | `gap`의 입력 (4번째 블록 최종 출력) | 256×14×14 |
| ResNet-50 | `layer4` 출력 | 2048×7×7 |
| ConvNeXt-Tiny | `stages[-1]` 출력 | 768×7×7 |

## 7. 개인정보

- Gradio는 브라우저에 그림을 보내려고 업로드 사진과 출력 그림을 임시 파일로 둡니다. 그 위치를 **이 앱 전용 새 임시 폴더**(`%TEMP%\face_age_gradio_xxxx`)로 바꿉니다.
- **1분마다** 확인해서 1분 지난 파일을 지웁니다 (`delete_cache=(60, 60)`).
- **앱을 끌 때(Ctrl+C) 폴더째 지웁니다.** 강제 종료로 남은 폴더는 다음에 켤 때 지웁니다.
- 앱 코드는 사진을 메모리로만 다루고, **Gradio 사용 통계 전송도 끕니다.**
- 기본은 **`127.0.0.1`(이 PC)에서만** 열립니다. `--lan`/`--share`일 때만 밖에서 접속됩니다 (5절 주의 참고).
- 앱은 1개만 켜세요. 2개를 동시에 켜면 나중 것이 먼저 것의 임시 폴더를 지웁니다.

## 8. 결과를 볼 때 주의할 점

- **얼굴 크기 지름길**: 학습 사진 중 AAF는 얼굴이 작게(머리·배경 포함) 찍혀 있고, 나이가 많은 나이대일수록 AAF 비율이 높습니다. 작은 CNN은 "얼굴이 작게 찍힌 사진 = 나이가 많다"는 단서를 쓰는 것으로 보여, 앱처럼 얼굴을 꽉 차게 자르면 동아시아 성인 얼굴이 **실제보다 어리게** 나올 수 있습니다. 사전학습 두 모델은 Grad-CAM에서 이런 징후가 약했지만(히트맵이 얼굴 가운데에 모임), 같은 다시 자르기 실험은 아직 돌리지 않았습니다 ([Grad-CAM 3모델 비교](../개발일지/LIM/2026-11-28_GradCAM_3모델비교.md)).
- 자세한 내용: [개발일지 17](../개발일지/LIM/2026-11-28_17_출처별_자르기차이_지름길확인.md), [개발일지 18](../개발일지/LIM/2026-11-28_18_AAF_다시자르기_실험.md). 예전 README의 여백·재자르기 수치표는 Haar 검출 기준이라 뺐습니다.
- **주목 비중 ≠ 영향력**: Grad-CAM이 많이 본 곳과 가렸을 때 확률이 가장 떨어지는 곳은 대부분 다릅니다 (val 9장에서 일치 1~2/9). 가림 실험은 학습 때 없던 블러 사진이라 수치가 과장될 수 있습니다 ([개발일지 20](../개발일지/LIM/2026-11-28_20_GradCAM_글설명.md) 4-4).
- 예측은 겉보기 나이 기준이며 참고용입니다.

### 모델 성능 참고 (시험 데이터 13,789장, macro-F1)

| 작은 CNN | ResNet-50 | ConvNeXt-Tiny |
|---:|---:|---:|
| 0.449 | 0.557 | **0.582** |

출처: [ResNet50/README.md 7절](../ResNet50/README.md#7-시험-데이터-최종-평가-2026-11-28), 작은 CNN은 [`작은CNN/results/small_cnn_v2_lr1e-3/eval_test/metrics.json`](../작은CNN/results/small_cnn_v2_lr1e-3/eval_test/metrics.json)과 같음

## 9. 발표 시연 · 남은 일

- **발표 3분 시연 시나리오**: [개발일지 Gradio 시연 6절](../개발일지/LIM/2026-11-28_Gradio시연.md#6-발표용-3분-시연-시나리오)
- **시연 중 실패 상황과 대처**: [개발일지 Gradio 시연 7절](../개발일지/LIM/2026-11-28_Gradio시연.md#7-시연-중-실패할-수-있는-상황과-대처)

**남은 일**

- [x] 얼굴 검출을 Haar → YuNet으로 교체 (새 앱 `작은CNN/app_gradio.py`)
- [x] 새 데이터셋(v2)으로 재학습한 작은 CNN 가중치로 교체
- [x] ResNet-50 · ConvNeXt v2 가중치로 세 모델 비교 · Grad-CAM 확인 (`--selftest`)
- [x] Grad-CAM 글 설명 (영역별 주목 비중 + 가림 실험)
- [x] 작은 CNN 기본 경로를 git에 있는 `small_cnn_best.pth`로 변경 (새 clone에서 바로 읽힘)
- [ ] 얼굴 크기 지름길: 재학습으로 줄이면(확대·축소 증강 또는 모든 출처를 같은 기준으로 다시 자르기) 새 가중치로 교체
- [ ] 실제 웹캠 사진 여러 장으로 정렬 켬/끔 비교
- [ ] AAF처럼 넓게 잘린 사진에 맞춘 여백도 필요한지 결정 (지금은 FairFace·UTKFace 기준)
- [ ] 주목 비중 1위 vs 가림 1위 비교를 val 전체로 (지금은 9장)
- [ ] 브라우저 첫 요청만 약 2초 걸리는 원인 확인
- [ ] `--selftest` ② 합성 사진 배경을 얼굴 없는 배경으로 바꿔 다시 재기

## 10. 예전 버전 (`시연/app.py`)

- OpenCV **Haar cascade**로 얼굴을 찾는 첫 시연 앱입니다. 지금은 쓰지 않으며 코드는 그대로 둡니다.
- Haar cascade는 OpenCV 5.x에 빠져 있어서 **OpenCV 4.x가 필요합니다.** 저장소 `.venv`(OpenCV 5.1)에서는 안 돌 수 있습니다.
- 쓰려면 `시연\requirements.txt`대로 **별도 가상환경**을 만들어야 합니다.
