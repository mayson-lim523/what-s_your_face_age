import os
import shutil

src = r"C:\Users\PC\Desktop\All-Age-Faces Dataset\original images"
dest = r"C:\Users\PC\Desktop\All-Age-Faces Dataset"

for filename in os.listdir(src):

    if not filename.lower().endswith((".jpg",".jpeg",".png")):
        continue

    try:
        name = os.path.splitext(filename)[0]

        age = int(name.split("A")[1])
       
        if age <= 19:
            age_group = "kid"

        elif age <= 64:
            age_group = "adult"

        else:
            age_group = "oldman"

        foldername = os.path.join(dest, age_group)

        if not os.path.exists(foldername):
            os.makedirs(foldername)

        source_file = os.path.join(src, filename)

        destination_file = os.path.join(foldername, filename)

        shutil.copy2(source_file, destination_file)

        print(filename, "->", age, "->", age_group)

    except (ValueError, IndexError):
        print("나이가 안나와있음 : ", filename)

print("AAF DONE")
