# =============================================================================
# 모듈 시작: make_splits_csv.py — 팀 공통 splits_csv 만들기 (한 번만 실행)
# =============================================================================
# 목적: data/training, data/validation, data/testing 폴더의 사진 목록을
#       splits_csv/train.csv, val.csv, test.csv 로 저장한다 (팀원 코드와 같은 형식).
# 열:   path(data 폴더 기준 상대 경로, "/" 사용), band(나이대 번호 0~6),
#       source, race_group(출처·인종 정보가 없어서 "unknown")
# 실행: 프로젝트 폴더에서  python make_splits_csv.py
# =============================================================================

import os                                               # 폴더 안 파일 목록 읽기, 경로 합치기
import re                                               # 정규식: 폴더 이름 앞의 숫자만 뽑기
import pandas as pd                                     # 표 만들기 → CSV 저장
from torchvision import datasets                        # ImageFolder: 검증용으로 같은 폴더를 다른 방법으로 읽어 비교

DATA_DIR = "data"                                       # 데이터 폴더 (프로젝트 폴더 기준 상대 경로)
OUT_DIR = "splits_csv"                                  # CSV 저장 폴더
# (CSV 이름, data 안 폴더 이름, 있어야 할 장수)
SPLITS = [("train", "training", 71735), ("val", "validation", 15288), ("test", "testing", 15289)]


def folder_to_band(name):                               # (name=나이대 폴더 이름) → 나이대 번호
    # re.match(r"\d+", 글자): 글자 맨 앞에서 연속된 숫자만 찾음 → '0 9'→'0', '20 29'→'20', '60over'→'60'
    # int(...) // 10: 정수로 바꾼 뒤 10으로 나눈 몫 → 0, 2, 6 / min(..., 6): 혹시 70 이상 폴더가 있어도 6으로
    return min(int(re.match(r"\d+", name).group()) // 10, 6)


def build_rows(split_dir):                              # (split_dir=예: data/training) → [(path, band), ...]
    rows = []                                           # 결과를 모을 빈 리스트
    for folder in sorted(os.listdir(split_dir)):        # 나이대 폴더를 이름 순서로 ('0 9', '10 19', ..., '60over')
        for fname in sorted(os.listdir(os.path.join(split_dir, folder))):   # 폴더 안 사진 파일 이름들 (정렬해서 매번 같은 순서)
            rows.append((f"{folder}/{fname}", folder_to_band(folder)))     # 경로는 "폴더/파일" 형식 (윈도우에서도 "/")
    return rows                                         # 사진 목록 반환


if __name__ == "__main__":
    for _, folder, _ in SPLITS:                         # testing 등 폴더가 하나라도 없으면 바로 멈춤
        if not os.path.isdir(os.path.join(DATA_DIR, folder)):
            raise SystemExit(f"{DATA_DIR}/{folder} 폴더가 없습니다. 멈춥니다.")   # 프로그램 종료 + 메시지
    os.makedirs(OUT_DIR, exist_ok=True)                 # (폴더, exist_ok=이미 있어도 OK) splits_csv 폴더 만들기

    for csv_name, folder, expected in SPLITS:           # train / val / test 차례로
        split_dir = os.path.join(DATA_DIR, folder)      # 예: data/training
        rows = build_rows(split_dir)                    # 사진 목록 만들기
        df = pd.DataFrame(rows, columns=["path", "band"])   # (행 리스트, columns=열 이름) 표로
        df["source"] = "unknown"                        # 출처 정보 없음 (팀 형식에 맞추려고 열만 둠)
        df["race_group"] = "unknown"                    # 인종 정보 없음
        df.to_csv(os.path.join(OUT_DIR, f"{csv_name}.csv"), index=False, encoding="utf-8")   # 저장 (행 번호 열 없이)

        # ---------- 검증 1: 장수 ----------
        ok_n = len(df) == expected                      # 있어야 할 장수와 같은지
        print(f"[{csv_name}] {len(df)}장 (기대 {expected}) → {'OK' if ok_n else '불일치'}")
        # ---------- 검증 2: 나이대별 장수 ----------
        # value_counts(): 값마다 개수 / sort_index(): 나이대 번호 순서로 정렬
        print("  나이대별:", df["band"].value_counts().sort_index().to_dict())
        # ---------- 검증 3: ImageFolder와 (경로, 라벨) 집합 비교 ----------
        # ImageFolder.samples = [(전체 경로, 라벨 번호), ...] — 사진을 열지 않고 목록만 만들어서 빠름
        # os.path.relpath(경로, 기준): 기준 폴더에서 본 상대 경로 → "20 29\\xxx.jpg" → "\\"를 "/"로 통일
        imf = datasets.ImageFolder(split_dir)
        imf_set = {(os.path.relpath(p, split_dir).replace("\\", "/"), lbl) for p, lbl in imf.samples}
        csv_set = set(zip(df["path"], df["band"]))      # CSV의 (path, band) 쌍 집합
        same = imf_set == csv_set                       # 집합이 완전히 같으면 True (빠진 것도, 라벨 다른 것도 없음)
        print(f"  ImageFolder와 집합 비교 → {'완전히 같음' if same else '다름'} "
              f"(CSV에만 {len(csv_set - imf_set)}개, ImageFolder에만 {len(imf_set - csv_set)}개)")
        if not (ok_n and same):                         # 하나라도 틀리면 멈춤
            raise SystemExit("검증 실패. 멈춥니다.")

# =============================================================================
# 모듈 끝: make_splits_csv.py
# =============================================================================
