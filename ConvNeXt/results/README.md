# 결과 CSV

| 파일 | 내용 | 출처 |
|---|---|---|
| `summary_val.csv` | 모든 실험의 검증 데이터 비교표 (정확도, 한 칸 차이 정확도, macro-F1). `v2_`로 시작하는 실험이 새 데이터(`division_224AFAD-X`) 결과이고, 나머지는 이전 데이터(`split_result_224`) 결과입니다. 두 데이터의 숫자는 검증 사진이 달라 직접 비교하면 안 됩니다. | `python step5_evaluate.py --summary results --tag val` 출력 (2026-11-28) |
| `v2_convnext_history.csv` | 새 데이터, 증강 있음 — 에폭별 기록 (원본 `results/v2_convnext/history.csv`) | 학습 코드가 저장한 원본 |
| `v2_noaug_convnext_history.csv` | 새 데이터, 증강 없음 — 에폭별 기록 (원본 `results/v2_noaug_convnext/history.csv`) | 학습 코드가 저장한 원본 |

- 모델 파일(`best.pt`, `last.pt`)은 용량이 커서 올리지 않습니다.

`history.csv` 열: `epoch`(에폭), `head_only`(출력층만 학습했는지), `train_loss`·`train_acc`(학습 손실·정확도), `val_loss`·`val_acc`(검증 손실·정확도), `val_off1_acc`(한 칸 차이 정확도), `val_macro_f1`, `lr`(학습률), `sec`(걸린 시간, 초).
