import os

folder = '/Users/gimgyuwon/Desktop/데이터 결과/split_result'

splits = ["training", "validation", "testing"]

counts = {}

for split in splits:
    split_path = os.path.join(folder, split)
    count = 0

    for age_group in os.listdir(split_path):
        age_path = os.path.join(split_path, age_group)

        if os.path.isdir(age_path):
            count += len([
                file for file in os.listdir(age_path)
                if file.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

    counts[split] = count


total = sum(counts.values())

print("전체 이미지 수:", total)
print()

print("===== 전체 분할 비율 =====")

for split in splits:
    ratio = counts[split] / total * 100

    print(
        split,
        ":",
        counts[split],
        "장 /",
        round(ratio, 2),
        "%"
    )


# ==========================================
# 여기부터 나이대별 분할 비율 확인
# ==========================================

print("\n===== 나이대별 분할 비율 =====")

age_groups = [
    "0 9",
    "10 19",
    "20 29",
    "30 39",
    "40 49",
    "50 59",
    "60over"
]

for age_group in age_groups:

    age_counts = {}
    age_total = 0

    for split in splits:

        age_path = os.path.join(
            folder,
            split,
            age_group
        )

        if os.path.isdir(age_path):

            count = len([
                file for file in os.listdir(age_path)
                if file.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

        else:
            count = 0

        age_counts[split] = count
        age_total += count


    print("\n", age_group, "- 전체:", age_total, "장")

    for split in splits:

        if age_total > 0:
            ratio = age_counts[split] / age_total * 100
        else:
            ratio = 0

        print(
            " ",
            split,
            ":",
            age_counts[split],
            "장 /",
            round(ratio, 2),
            "%"
        )
