# ResNet-50 / ConvNeXt-Tiny 학습 진행 기록

전처리된 얼굴 이미지(224×224)로 **ResNet-50**과 **ConvNeXt-Tiny**를 학습해 7개 나이대(0–9세, 10대, 20대, 30대, 40대, 50대, 60세 이상)를 분류하는 작업의 진행 기록입니다.

- 담당: 모델 학습 (ResNet-50, ConvNeXt-Tiny)
- 기록일: 2026-11-09
- 현재 상태: **ResNet-50 학습·검증 평가 완료** (8에폭 모델, 12에폭에서 조기 종료). ConvNeXt-Tiny 결과와 두 모델 비교는 [`../ConvNeXt`](../ConvNeXt) 참고

### 결과 요약 (검증 데이터 15,288장)

| 정확도 | 한 칸 차이 정확도 | macro-F1 | 가장 좋은 에폭 |
|---:|---:|---:|---:|
| 61.9% | 92.9% | 0.615 | 8 |

### 모델 역할

- **학습 방식**: 사전학습 모델 미세조정 (PyTorch + timm)
- **역할**: 기준 모델 — 비교 실험의 기본값
- **비교 실험**: 이 모델에서 조건을 한 가지씩만 바꿔서 비교합니다.
  - 모델 구조: 작은 CNN / ResNet-50 / ConvNeXt-Tiny → ConvNeXt-Tiny와의 비교는 [`../ConvNeXt`](../ConvNeXt)
  - 학습 방법: 마지막 층만 학습 / 전체 미세조정
  - 데이터 전처리: 증강 적용 / 미적용, 보강 데이터 포함 / 미포함
  - 하이퍼파라미터: 학습률 0.001 / 0.0001 / 0.00001

---

## 1. 실행 환경

| 항목 | 내용 |
|---|---|
| 컴퓨터 | 데스크탑, NVIDIA GeForce RTX 3070 (메모리 8GB) |
| 운영체제 | Windows |
| 파이썬 | Python 3.11, 가상환경(`venv`) 사용 |
| 주요 라이브러리 | PyTorch(GPU 버전), torchvision, timm, pandas, scikit-learn, matplotlib, tqdm, pillow, grad-cam |
| 작업 폴더 | `C:\age_project` |

작업 폴더 구성:

```
C:\age_project\
├─ venv\                 가상환경
├─ split_result_224\     전처리 담당 팀원이 만든 분리 데이터 (training / validation / testing)
├─ splits_csv\           학습 코드용 CSV (make_split_csv.py로 생성)
├─ results\              학습 결과 (실험별 폴더)
├─ common.py
├─ make_split_csv.py
├─ step4_train.py
├─ step5_evaluate.py
└─ run_models.bat
```

사용한 코드는 이 폴더의 [`code/`](code/)에 있습니다.

---

## 2. 진행 과정

### 2-1. 가상환경 만들기와 라이브러리 설치

```
cd C:\
mkdir age_project
cd age_project
python -m venv venv
venv\Scripts\activate
pip install timm pandas scikit-learn matplotlib tqdm pillow grad-cam
```

![라이브러리 설치](images/01_pip_install.png)

> **문제:** 설치 도중 `Found existing installation: typing_extensions` 이후 `Successfully installed` 문구 없이 끝났다. cmd에서 Ctrl+C를 눌러 설치가 중단된 것으로 확인했다(cmd에서 Ctrl+C는 복사가 아니라 실행 중지).
> **해결:** 같은 설치 명령을 다시 실행했다.

### 2-2. 팀원이 나눈 데이터 확인

전처리 담당 팀원이 나눈 `split_result_224` 폴더(training / validation / testing)를 그대로 사용하기로 했다. 팀 전체가 같은 분리를 써야 실험끼리 비교할 수 있기 때문이다.

```
dir split_result_224\training
```

![분리 폴더 구조](images/02_split_folder_structure.png)

- 각 묶음 안에 나이대별 폴더 7개가 있다: `0 9`, `10 19`, `20 29`, `30 39`, `40 49`, `50 59`, `60over`
- 학습 코드는 "사진 경로 + 나이대 번호" CSV를 읽으므로, 폴더 구조를 CSV로 바꾸는 `make_split_csv.py`를 만들었다.
- 처음 버전은 `10 19`처럼 숫자 사이에 띄어쓰기가 있는 폴더 이름을 잘못 읽는 문제가 있어 수정했다(앞 숫자만 읽도록 변경).

### 2-3. 오류 1: pandas 없음

![pandas 오류](images/03_error_no_pandas.png)

> **원인:** 2-1의 설치가 중간에 끊겨 pandas가 설치되지 않았다.
> **해결:** 라이브러리 설치 명령을 다시 실행했다.

### 2-4. 오류 2: GPU 인식 안 됨 (`cuda` False)

```
python -c "import pandas, timm, sklearn, torch; print('OK', torch.cuda.is_available())"
```

![cuda False](images/04_cuda_false.png)

> **원인:** timm, grad-cam을 설치할 때 **CPU 전용 PyTorch**가 함께 설치됐다. 이 상태로는 GPU를 쓰지 못한다.
> **해결:** CPU용 torch를 지우고(`pip uninstall -y torch torchvision`), pytorch.org의 설치 명령으로 GPU(CUDA)용 PyTorch를 설치했다.

![PyTorch 설치 선택 화면](images/05_pytorch_install_selector.png)

- `nvidia-smi` 기준 그래픽 드라이버가 지원하는 CUDA 버전은 **12.8**이었다.
- pytorch.org 기본값은 **CUDA 13.2용**(안정 버전 2.15.0) 빌드다. 드라이버가 지원하는 버전보다 높아서, 드라이버 업데이트 또는 CUDA 12.8용(cu128) 빌드 설치가 필요하다고 판단했다.
- 결과적으로 **PyTorch 2.15.0(CUDA 13.2용)을 그대로 사용**했고, 별도 조치 없이 `torch.cuda.is_available()`이 `True`로 확인됐다. 설치된 드라이버가 이미 이 버전을 지원한 것으로 보인다.

![torch 2.15.0+cu132 True](images/05b_torch_cu132_cuda_true.png)

### 2-5. 오류 3: common.py 없음

![common 오류](images/06_error_no_common.png)

> **원인:** `make_split_csv.py`가 같은 폴더의 `common.py`(나이대 이름 정의)를 불러오는데, 그 파일이 작업 폴더에 없었다.
> **해결:** `common.py`, `step4_train.py`, `step5_evaluate.py`를 `C:\age_project`에 넣었다.

### 2-6. CSV 만들기 결과

```
python make_split_csv.py --root split_result_224 --out-dir splits_csv
```

![분리 결과](images/07_split_csv_counts.png)

| 나이대 | 학습 | 검증 | 시험 |
|---|---:|---:|---:|
| 0–9세 | 11,561 | 2,474 | 2,470 |
| 10대 | 12,000 | 2,571 | 2,571 |
| 20대 | 12,000 | 2,571 | 2,571 |
| 30대 | 12,000 | 2,571 | 2,571 |
| 40대 | 12,104 | 2,468 | 2,545 |
| 50대 | 6,801 | 1,433 | 1,453 |
| 60세 이상 | 5,269 | 1,200 | 1,108 |
| **합계** | **71,735** | **15,288** | **15,289** |

- 전체 102,312장, 비율 **70 : 15 : 15**
- 10~30대는 학습 데이터에서 12,000장으로 상한이 걸려 있다.
- **50대와 60세 이상은 다른 나이대의 절반 정도로 적다.** → 학습 시 손실 가중치로 보정(2-7)

### 2-7. 시험 학습 (2에폭)

전체 흐름이 끝까지 돌아가는지 확인하기 위해 2에폭만 짧게 학습했다.

```
python step4_train.py --name smoke_test --model resnet50 --train-csv splits_csv/train.csv --val-csv splits_csv/val.csv --image-root split_result_224 --epochs 2
```

![사전학습 가중치 다운로드와 손실 가중치](images/08_pretrained_download_classweight.png)

- `장치: cuda` → GPU로 학습 확인
- ResNet-50 사전학습 가중치(`resnet50.a1_in1k`, 약 102MB)를 Hugging Face에서 내려받음
- HF_TOKEN, symlink 관련 경고는 다운로드 방식에 대한 안내로, 학습에는 영향 없음
- **나이대별 손실 가중치:** `[0.89 0.85 0.85 0.85 0.85 1.51 1.94]` → 사진이 적은 50대(1.51), 60세 이상(1.94)에 더 큰 벌점

![시험 학습 결과](images/09_smoke_test_result.png)

| 에폭 | 학습 손실 | 학습 정확도 | 검증 손실 | 검증 정확도 | 한 칸 차이 정확도 | macro-F1 | 시간 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.799 | 0.290 | 1.658 | 0.355 | 0.621 | 0.303 | 233초 |
| 2 | 1.315 | 0.458 | 1.222 | 0.493 | 0.810 | 0.484 | 501초 |

- 1에폭은 몸통을 얼리고 마지막 층만 학습해서 빠르고(233초), 2에폭부터 전체를 학습해 약 2배 걸린다(501초).
- 학습·검증 손실이 함께 줄어 아직 과적합은 보이지 않는다.

### 2-8. 본 학습

`run_models.bat`로 ResNet-50 → ConvNeXt-Tiny를 순서대로 20에폭씩 학습하고, 끝나면 검증 데이터로 평가와 비교표 생성까지 자동으로 진행한다.

```
set CMN=--train-csv splits_csv/train.csv --val-csv splits_csv/val.csv --image-root split_result_224 --epochs 20
python step4_train.py --name base_resnet50 --model resnet50 %CMN%
python step4_train.py --name m_convnext --model convnext_tiny %CMN%
```

![ResNet-50 본 학습](images/10_base_resnet50_training.png)

| 에폭 | 학습 손실 | 학습 정확도 | 검증 손실 | 검증 정확도 | 한 칸 차이 정확도 | macro-F1 | 시간 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1/20 | 1.799 | 0.290 | 1.658 | 0.355 | 0.621 | 0.303 | 221초 |
| 2/20 | 1.281 | 0.469 | 1.132 | 0.529 | 0.832 | 0.505 | 490초 |

- 2에폭 점수가 시험 학습보다 높은 이유: 20에폭 기준으로 학습률을 천천히 줄이는 설정이라, 2에폭 시점의 학습률이 시험 학습(2에폭 기준)보다 높게 유지된다.
### 2-9. 조기 종료

![ResNet-50 조기 종료](images/11_resnet50_early_stop.png)

| 에폭 | 학습 손실 | 학습 정확도 | 검증 손실 | 검증 정확도 | 한 칸 차이 정확도 | macro-F1 | 시간 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 (최고) | 0.757 | 0.670 | **0.893** | **0.619** | 0.929 | **0.615** | 485초 |
| 9 | 0.723 | 0.685 | 0.913 | 0.608 | 0.929 | 0.603 | 589초 |
| 10 | 0.690 | 0.699 | 0.919 | 0.616 | 0.929 | 0.612 | 333초 |
| 11 | 0.657 | 0.713 | 0.937 | 0.613 | 0.928 | 0.607 | 297초 |
| 12 | 0.627 | 0.726 | 0.947 | 0.614 | 0.929 | 0.609 | 749초 |

- 8에폭 이후 **학습 손실은 계속 줄지만(0.757 → 0.627) 검증 손실은 늘어난다(0.893 → 0.947).** 학습 사진을 외우기 시작한 과적합 신호다.
- 검증 macro-F1이 4에폭 연속 오르지 않아 12에폭에서 조기 종료됐고, 저장된 모델은 **8에폭 모델**이다.
- 에폭 시간(297~749초)이 들쭉날쭉한 것은 학습 중 컴퓨터의 다른 작업이나 디스크 읽기 속도 영향으로 보이며, 결과에는 영향이 없다.

### 2-10. 검증 데이터 평가

```
python step5_evaluate.py --exp-dir results/base_resnet50 --split-csv splits_csv/val.csv --image-root split_result_224 --tag val
python step5_evaluate.py --summary results --tag val
```

![평가 결과](images/12_eval_val_results.png)

![비교표](images/13_summary_table.png)

---

## 3. 결과 해석

### 3-1. 혼동 행렬

![ResNet-50 혼동 행렬](images/14_resnet50_confusion_matrix.png)

| 실제 나이대 | 맞힌 비율 | 가장 많이 헷갈린 나이대 |
|---|---:|---|
| 0–9세 | 91% | 10대 (6%) |
| 10대 | 75% | 20대 (11%), 0–9세 (9%) |
| 20대 | 58% | 30대 (27%) |
| 30대 | 47% | 20대 (28%), 40대 (18%) |
| 40대 | 45% | 30대 (25%), 50대 (18%) |
| 50대 | 48% | 40대 (26%), 60세 이상 (18%) |
| 60세 이상 | 66% | 50대 (26%) |

- **양 끝은 쉽고 가운데는 어렵다.** 0–9세(91%)와 60세 이상(66%)은 잘 맞히지만 30~50대는 45~48%에 그친다.
- **데이터가 가장 많은 30·40대(각 12,000장)가 가장 낮다.** 사진 수가 가장 적은 60세 이상(5,269장)이 오히려 66%다. 손실 가중치로 적은 나이대를 보정했고, 성능은 데이터 양보다 **이웃 나이대와 생김새가 얼마나 비슷한지**에 더 크게 좌우된 것으로 보인다.
- **오답은 대부분 바로 옆 나이대다.** 두 칸 이상 떨어진 칸은 대부분 0~2%이고, 그래서 한 칸 차이 정확도가 92.9%로 높다.
- 20대는 10대(8%)보다 30대(27%)로 훨씬 많이 틀린다. 데이터셋마다 나이 기준(실제 나이 / 보이는 나이)이 다른 점 등이 원인일 수 있다.

### 3-2. 학습 곡선

![ResNet-50 학습 곡선](images/15_resnet50_learning_curve.png)

- 4에폭까지 학습·검증 손실이 함께 빠르게 줄어든다.
- 그 뒤 검증 손실은 0.9 근처에서 평평하고, 학습 손실만 계속 내려간다. 검증 정확도도 4에폭 이후 60~62%에서 정체된다.
- 과적합은 **완만한 편**이다(12에폭 기준 학습 72.6%, 검증 61.4%, 차이 약 11%p).
- 1에폭에서 검증 정확도가 학습 정확도보다 높은 것은, 학습 정확도가 에폭 동안 점점 나아지는 모델의 평균이고 증강된(일부러 어렵게 바꾼) 사진으로 쟀기 때문이다.

---

## 4. 학습 설정 (두 모델 공통)

| 항목 | 값 |
|---|---|
| 입력 크기 | 224 × 224 |
| 출력 | 7개 나이대 |
| 사전학습 | ImageNet (timm) |
| 배치 크기 | 64 |
| 최적화 | AdamW (weight decay 0.05) |
| 학습률 | 몸통 0.0001, 새 출력층 0.001 |
| 학습률 스케줄 | 1에폭 워밍업 후 코사인 감소 |
| 2단계 학습 | 1에폭은 출력층만 → 이후 전체 미세조정 |
| 최대 에폭 | 20 |
| 조기 종료 | 검증 macro-F1이 4에폭 연속 오르지 않으면 종료 |
| 손실 함수 | 교차 엔트로피 + 나이대별 손실 가중치 |
| 증강 (학습 데이터만) | 살짝 확대·자르기(90~100%), 좌우 반전, ±10° 회전, 밝기·대비 ±20% |
| 정규화 | ImageNet 평균·표준편차 |
| 혼합 정밀도 | 사용 (AMP) |

---

## 5. 해결한 문제 요약

| 문제 | 원인 | 해결 |
|---|---|---|
| 설치가 중간에 끝남 | cmd에서 Ctrl+C로 실행 중지 | 설치 명령 재실행, cmd에서는 Ctrl+C 사용 금지 |
| `No module named 'pandas'` | 설치 중단 | 라이브러리 재설치 |
| `cuda` False | CPU용 PyTorch가 설치됨 | GPU용 PyTorch 재설치 |
| `No module named 'common'` | 작업 폴더에 common.py 없음 | 코드 파일을 작업 폴더에 배치 |
| 폴더 이름 `10 19`를 잘못 읽음 | 띄어쓰기를 지우고 숫자를 붙여 읽음 | 앞 숫자만 읽도록 make_split_csv.py 수정 |

---

## 6. 다음 할 일

- [x] ResNet-50 본 학습 완료 (`results/base_resnet50/`)
- [x] ConvNeXt-Tiny 본 학습 완료 (`results/m_convnext/`) → [`../ConvNeXt`](../ConvNeXt)
- [x] 검증 데이터 평가: 학습 곡선, 혼동 행렬 (`eval_val/`)
- [x] 두 모델 비교표 (`results/summary_val.csv`) 정리 및 해석
- [ ] 비교 결과로 최종 모델 결정 → 시험 데이터(test)로 한 번만 평가
- [ ] Grad-CAM으로 판단 근거 확인

## 참고

- 출처 정보 없음: `split_result_224`에는 사진별 출처(FairFace, AAF 등) 정보가 없어 출처별 성능 비교(`per_group.csv`)는 `unknown`으로만 나온다.
- 시험 데이터(testing)는 최종 모델 평가 전까지 사용하지 않는다.
