@echo off
chcp 65001 >nul
REM 새 데이터(division_224AFAD-X)로 ResNet-50 증강 있음/없음 학습. 조기 종료 없이 20에폭
REM 먼저 한 번: xcopy /E /I /Y "C:\Users\rhddy\Downloads\division_224AFAD-X" data_v2  /  python make_split_csv.py --root data_v2 --out-dir splits_csv_v2
cd /d C:\age_project
call venv\Scripts\activate
set D=--train-csv splits_csv_v2/train.csv --val-csv splits_csv_v2/val.csv --image-root data_v2 --epochs 20 --patience 100
set E=--split-csv splits_csv_v2/val.csv --image-root data_v2 --tag val
python step4_train.py --name v2_resnet50 --model resnet50 %D%
python step4_train.py --name v2_noaug_resnet50 --model resnet50 --no-augment %D%
python step5_evaluate.py --exp-dir results/v2_resnet50 %E%
python step5_evaluate.py --exp-dir results/v2_noaug_resnet50 %E%
python step5_evaluate.py --summary results --tag val
pause
