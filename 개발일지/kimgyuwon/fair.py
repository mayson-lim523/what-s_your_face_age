import shutil, os
import pandas as pd

#fairface의 경우는 val과 train이 이미 나뉘어 있었음. 이에 따라서 둘다 분류하는 방식으로 진행

#폴더 경로 지정
source_folder = r"C:\Users\PC\Desktop\FairFace\val"
output_folder = r"C:\Users\PC\Desktop\FairFace"

#CSV파일 내의 age라벨 읽기
labels = pd.read_csv(r"C:\Users\PC\Desktop\FairFace/val_labels.csv")
labels = labels.sort_values('age')

#class name분류
class_names = list(labels.age.unique())


train_images = '\val'
train_cat = '\val_'

#파일 생성
for i in class_names:
    os.makedirs(os.path.join(output_folder,i),
                exist_ok=True)

#클래스 이름을 읽음
for c in class_names:
    for i in list(labels[labels['age']==c]['file']):
        filename = os.path.basename(i)                                  #파일이름 설정
        get_image = os.path.join(source_folder,filename)                #입력 파일에서 이미지 읽
        move_image_to_cat = shutil.copy2(get_image,                     #출력 파일에 이미지 복사
                                        os.path.join(output_folder,c))

#fair의 경우는 이전 코드와 다른게 없음.
#CSV라벨이 연령별로 분류되어있는데, 다만 60-69세의 경우는 라벨을 바꾸지 않는 이상 분류가 힘들기 때문에, 60-69세의 경우는 3클래스 파일에서 제외함.
