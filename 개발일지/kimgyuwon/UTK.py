import os
import shutil

#경로 설정
#1-1. 이미지 파일 설정
source_folder = r"C:\Users\PC\Desktop\UTK\UTKFace"
#1-2. 출력 파일 설정
output_folder = r"C:\Users\PC\Desktop\UTK"

#2-1. 파일 클래스 설정(3클래스로 변경)
age_groups = [
    "kid",
    "adult",
    "oldman",
]

#2-2 출력 파일 내부에 폴더 형성 파일
for group in age_groups:
    os.makedirs(
        os.path.join(output_folder, group),
        exist_ok=True
    )

#3-1. 파일 불러오기
for filename in os.listdir(source_folder):

    #확장자 확인(없는 경우 인식 불가)
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    #이미지 이름으로 분류(UTK의 경우는 _로 구분)
    try:
        age = int(filename.split("_")[0])

        #if else문으로 나이대 별로 구분
        if age <= 19:
            age_group = "kid"
            
        elif age <= 64:
            age_group = "adult"
            
        else:
            age_group = "oldman"

        #UTK데이터셋 파일 경로 지정
        source_path = os.path.join(
            source_folder,
            filename
        )

        #복사될 이미지 파일 경로 지정
        destination_path = os.path.join(
            output_folder,
            age_group,
            filename
        )
        
        #이미지 파일 복사
        shutil.copy2(
            source_path,
            destination_path
        )

#4-1. 이미지 분류시 출력되는 print 문
        print(filename, "->", age_group)

    except ValueError:
        print("나이가 나와있지 안음 : ", filename)

print("UTKFace DONE")
