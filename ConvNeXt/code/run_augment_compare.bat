@echo off
chcp 65001 >nul
REM ConvNeXt-Tiny (ImageNet-1k 가중치) 사전학습 조건 통일 + 증강 유무 비교
cd /d C:\age_project
call venv\Scripts\activate
REM 1) 증강 있음: ResNet-50과 같은 ImageNet-1k 가중치 (조기 종료 사용)
python step4_train.py --name m_convnext_in1k --model convnext_tiny.fb_in1k --train-csv splits_csv/train.csv --val-csv splits_csv/val.csv --image-root split_result_224 --epochs 20
python step5_evaluate.py --exp-dir results/m_convnext_in1k --split-csv splits_csv/val.csv --image-root split_result_224 --tag val
REM 2) 증강 없음: 증강만 끄고, 조기 종료 없이 20에폭
python step4_train.py --name noaug_convnext --model convnext_tiny.fb_in1k --no-augment --patience 100 --train-csv splits_csv/train.csv --val-csv splits_csv/val.csv --image-root split_result_224 --epochs 20
python step5_evaluate.py --exp-dir results/noaug_convnext --split-csv splits_csv/val.csv --image-root split_result_224 --tag val
python plot_history_compare.py --a results/m_convnext_in1k --b results/noaug_convnext --label-a "증강 있음" --label-b "증강 없음" --title "ConvNeXt-Tiny (ImageNet-1k): 증강 있음 vs 없음" --out aug_compare_convnext.png
pause
