import os
import shutil

source_folder = r"C:\Users\PC\Desktop\UTK\UTKFace"
output_folder = r"C:\Users\PC\Desktop\UTK"

age_groups = [
    "kid",
    "adult",
    "oldman",
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

        if age <= 19:
            age_group = "kid"
            
        elif age <= 64:
            age_group = "adult"
            
        else:
            age_group = "oldman"

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
        print("나이가 나와있지 안음 : ", filename)

print("UTKFace DONE")
