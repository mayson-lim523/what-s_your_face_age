import os
import shutil

source_folder = r"C:\Users\PC\Desktop\UTK\UTKFace"
output_folder = r"C:\Users\PC\Desktop\UTK\classified"

age_groups = [
    "0_9",
    "10s",
    "20s",
    "30s",
    "40s",
    "50s",
    "60plus"
]

for group in age_groups:
    os.makedirs(
        os.path.join(output_folder, group),
        exist_ok=True
    )

for filename in os.listdir(source_folder):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    try:
        age = int(filename.split("_")[0])

        if age <= 9:
            age_group = "0_9"
        elif age <= 19:
            age_group = "10s"
        elif age <= 29:
            age_group = "20s"
        elif age <= 39:
            age_group = "30s"
        elif age <= 49:
            age_group = "40s"
        elif age <= 59:
            age_group = "50s"
        else:
            age_group = "60plus"

        source_path = os.path.join(
            source_folder,
            filename
        )

        destination_path = os.path.join(
            output_folder,
            age_group,
            filename
        )

        shutil.copy2(
            source_path,
            destination_path
        )

        print(filename, "->", age_group)

    except ValueError:
        print("나이를 읽지 못한 파일:", filename)

print("UTKFace 분류 완료!")
