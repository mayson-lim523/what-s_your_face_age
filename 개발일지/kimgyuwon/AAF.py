import os
import shutil

#경로 설정
#1-1. 이미지 파일 설정
src = r"C:\Users\PC\Desktop\All-Age-Faces Dataset\original images"
dest = r"C:\Users\PC\Desktop\All-Age-Faces Dataset"

#2-1. 파일 불러오기
for filename in os.listdir(src):

    #확장자 확인
    if not filename.lower().endswith((".jpg",".jpeg",".png")):
        continue

    #이미지 이름으로 분류(AAF의 경우는 A로 구분)
    try:
        name = os.path.splitext(filename)[0]
        
        age = int(name.split("A")[1])

        #if else문으로 나이대별로 구분
        if age <= 19:
            age_group = "kid"

        elif age <= 64:
            age_group = "adult"

        else:
            age_group = "oldman"
        
        #파일 이름 설
        foldername = os.path.join(dest, age_group)

        #지정된 출력 경로에 파일이 없을 경우 파일 생성
        if not os.path.exists(foldername):
            os.makedirs(foldername)

        #이미지 데이터 경로
        source_file = os.path.join(src, filename)

        #복사될 이미지 경로
        destination_file = os.path.join(foldername, filename)

        #파일 복사
        shutil.copy2(source_file, destination_file)

        print(filename, "->", age, "->", age_group)
        
    #3-1. 이미지 분류시 출력
    except (ValueError, IndexError):
        print("나이가 안나와있음 : ", filename)

print("AAF DONE")
