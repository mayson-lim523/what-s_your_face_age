@echo off
chcp 65001 >nul
REM ResNet-50 증강 유무 비교: 증강만 끄고(--no-augment) 나머지 조건은 그대로, 조기 종료 없이 20에폭
cd /d C:\age_project
call venv\Scripts\activate
python step4_train.py --name noaug_resnet50 --model resnet50 --no-augment --patience 100 --train-csv splits_csv/train.csv --val-csv splits_csv/val.csv --image-root split_result_224 --epochs 20
python step5_evaluate.py --exp-dir results/noaug_resnet50 --split-csv splits_csv/val.csv --image-root split_result_224 --tag val
REM 비교 상대: 증강 있음 20에폭 (results/full20_resnet50)
python plot_history_compare.py --a results/full20_resnet50 --b results/noaug_resnet50 --label-a "증강 있음" --label-b "증강 없음" --title "ResNet-50: 증강 있음 vs 없음" --out aug_compare_resnet50.png
pause
