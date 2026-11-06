import os
import shutil

src = "/Users/gimgyuwon/Desktop/data/All-Age-Faces Dataset/original images"
dest ='/Users/gimgyuwon/Desktop/data/All-Age-Faces Dataset/AAFclass'

for filename in os.listdir(src):

    if not filename.lower().endswith((".jpg",".jpeg",".png")):
        continue

    try:
        name = os.path.splitext(filename)[0]

        age = int(name.split("A")[1])
       
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

        foldername = os.path.join(dest, age_group)

        if not os.path.exists(foldername):
            os.makedirs(foldername)

        source_file = os.path.join(src, filename)

        destination_file = os.path.join(foldername, filename)

        shutil.copy2(source_file, destination_file)

        print(filename, "->", age, "->", age_group)

    except (ValueError, IndexError):
        print("나이를 못읽음:", filename)

print("DONE")
