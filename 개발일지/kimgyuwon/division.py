
import random
from pathlib import Path
from shutil import copy2

# ==========================================
# 1. 기본 경로 및 분할 비율 설정
# ==========================================

# kid / adult / oldman 폴더가 들어 있는 경로
IMAGE_DIR = Path(r"C:\Users\PC\Desktop\dataset")

# 분할한 데이터를 저장할 새 폴더
OUTPUT_DIR = Path(r"C:\Users\PC\Desktop\division")

CLASS_NAMES = ["kid", "adult", "oldman"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 1337

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ==========================================
# 2. 이미지 목록 생성 및 분할
# ==========================================

def create_image_lists(image_dir):

    image_lists = {}

    for class_name in CLASS_NAMES:

        class_dir = image_dir / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"클래스 폴더가 없습니다: {class_dir}"
            )

        # 폴더에서 이미지 파일만 가져오기
        file_list = sorted(
            file for file in class_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in EXTENSIONS
        )

        if len(file_list) == 0:
            raise ValueError(
                f"{class_name} 폴더에 이미지가 없습니다."
            )

        # 무작위 분할 (동일한 seed면 같은 결과)
        rng = random.Random(f"{SEED}_{class_name}")
        rng.shuffle(file_list)

        total = len(file_list)

        train_count = round(total * TRAIN_RATIO)
        val_count = round(total * VAL_RATIO)

        training_images = file_list[:train_count]

        validation_images = file_list[
            train_count:train_count + val_count
        ]

        testing_images = file_list[
            train_count + val_count:
        ]

        image_lists[class_name] = {
            "training": training_images,
            "validation": validation_images,
            "testing": testing_images
        }

        print(f"\n[{class_name}]")
        print(f"전체 이미지: {total}장")
        print(f"Train: {len(training_images)}장")
        print(f"Validation: {len(validation_images)}장")
        print(f"Test: {len(testing_images)}장")

    return image_lists


# ==========================================
# 3. 이미지를 각 폴더에 복사
# ==========================================

def divide_images():

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"원본 데이터 폴더가 없습니다: {IMAGE_DIR}"
        )

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"출력 폴더가 이미 존재합니다: {OUTPUT_DIR}\n"
            "기존 결과가 섞이지 않도록 새 출력 경로를 사용하세요."
        )

    # 이미지 목록 먼저 분할
    image_lists = create_image_lists(IMAGE_DIR)

    # 저장 폴더 생성
    OUTPUT_DIR.mkdir(parents=True)

    for split_name in ["training", "validation", "testing"]:

        for class_name in CLASS_NAMES:

            target_dir = OUTPUT_DIR / split_name / class_name

            target_dir.mkdir(parents=True, exist_ok=True)

            # 해당 분할의 이미지 복사
            for image_path in image_lists[class_name][split_name]:

                destination = target_dir / image_path.name

                copy2(image_path, destination)

    print("\n===================================")
    print("Train / Validation / Test 분할 완료!")
    print("비율: 70% / 15% / 15%")
    print(f"저장 위치: {OUTPUT_DIR}")
    print("===================================")


# ==========================================
# 4. 코드 실행
# ==========================================

if __name__ == "__main__":

    divide_images()
