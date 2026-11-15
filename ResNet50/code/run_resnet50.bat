cd /d C:\age_project
call venv\Scripts\activate
python step4_train.py --name base_resnet50 --model resnet50 --train-csv splits_csv/train.csv --val-csv splits_csv/val.csv --image-root split_result_224 --epochs 20
python step5_evaluate.py --exp-dir results/base_resnet50 --split-csv splits_csv/val.csv --image-root split_result_224 --tag val
pause
