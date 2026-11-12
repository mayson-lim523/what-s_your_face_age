# 작은 CNN 본 학습 결과 (2026-11-12)

| 파일 | 내용 |
|---|---|
| `small_cnn_best.pth` | 검증 macro-F1이 가장 좋았던 **18 에폭** 모델 가중치 (`model.state_dict()`) |
| `small_cnn_history.csv` | 20 에폭 학습 기록 (팀원 `history.csv`와 같은 열 이름) |
| `learning_curve.png` | 학습 곡선 (loss, accuracy) |
| `pred_samples.png` | 검증 데이터 무작위 8장 예측 (18 에폭 모델) |
| `CNN_4layers_used.py` | 이 결과를 만든 코드 그대로 (`USE_SUBSET = False`) |

## 학습 조건

- 데이터: `split_result_224` training 71,735장 / validation 15,288장 (팀원과 동일)
- 모델: SmallCNN (Conv 블록 4개, 32→64→128→256, GAP, Dropout 0.3), 파라미터 390,695개, 사전학습 없음
- 증강: RandomResizedCrop(0.9~1.0) + 좌우 반전 + ±10° 회전 + 밝기·대비 ±0.2 (팀원과 동일)
- 손실: CrossEntropyLoss + 나이대별 가중치 (팀원과 동일)
- 최적화: Adam, lr 0.001 (고정), 배치 64, 20 에폭, 시드 42
- 최고 모델 기준: 검증 macro-F1
- 학습 시간: 6,603초 (약 1시간 50분), 에폭당 약 320초

## 최고 성적 (18 에폭, 검증 15,288장)

| 정확도 | 한 칸 차이 정확도 | macro-F1 |
|---:|---:|---:|
| 50.1% | 82.9% | 0.496 |

## 불러오는 방법

```python
model = SmallCNN().to(device)
model.load_state_dict(torch.load("results/small_cnn_main/small_cnn_best.pth"))
model.eval()
```
