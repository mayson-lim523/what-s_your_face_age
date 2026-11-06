import shutil, os
import pandas as pd

source_folder = "/Users/gimgyuwon/Desktop/data/FairFace/train"
output_folder = "/Users/gimgyuwon/Desktop/data/FairFace/fairface_file"

labels = pd.read_csv("/Users/gimgyuwon/Desktop/data/FairFace/train_labels.csv")
labels = labels.sort_values('age')

class_names = list(labels.age.unique())

train_images = '\train'
train_cat = '\train_'

for i in class_names:
    os.makedirs(os.path.join("/Users/gimgyuwon/Desktop/data/FairFace/fairface_file",i),
                exist_ok=True)

for c in class_names:
    for i in list(labels[labels['age']==c]['file']):
        filename = os.path.basename(i)
        get_image = os.path.join("/Users/gimgyuwon/Desktop/data/FairFace/train",filename)
        move_image_to_cat = shutil.move(get_image,
                                        os.path.join("/Users/gimgyuwon/Desktop/data/FairFace/fairface_file",c))
