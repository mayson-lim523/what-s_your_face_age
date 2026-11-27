# 결과 CSV

| 파일 | 내용 | 출처 |
|---|---|---|
| `summary_val.csv` | 모든 실험의 검증 데이터 비교표 (정확도, 한 칸 차이 정확도, macro-F1). `v2_`로 시작하는 실험이 새 데이터(`division_224AFAD-X`) 결과이고, 나머지는 이전 데이터(`split_result_224`) 결과입니다. 두 데이터의 숫자는 검증 사진이 달라 직접 비교하면 안 됩니다. | `python step5_evaluate.py --summary results --tag val` 출력 (2026-11-28) |
| `v2_epoch_log_from_screenshots.csv` | 새 데이터 실험의 에폭별 기록 (학습 정확도, 검증 정확도, 검증 손실, 검증 macro-F1) | 학습 화면·`history.csv` **캡처에서 옮겨 적은 값** (소수점 셋째 자리). 원본 `history.csv`를 받으면 그 파일로 바꿀 예정입니다. |

- 증강 있는 ConvNeXt(`v2_convnext`)의 에폭별 기록은 아직 없습니다 (평가 결과만 `summary_val.csv`에 있음).
- 모델 파일(`best.pt`, `last.pt`)은 용량이 커서 올리지 않습니다.
