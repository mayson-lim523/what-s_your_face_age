import shutil, os
import pandas as pd

source_folder = r"C:\Users\PC\Desktop\FairFace\val"
output_folder = r"C:\Users\PC\Desktop\FairFace"

labels = pd.read_csv(r"C:\Users\PC\Desktop\FairFace/val_labels.csv")
labels = labels.sort_values('age')

class_names = list(labels.age.unique())

train_images = '\val'
train_cat = '\val_'

for i in class_names:
    os.makedirs(os.path.join(output_folder,i),
                exist_ok=True)

for c in class_names:
    for i in list(labels[labels['age']==c]['file']):
        filename = os.path.basename(i)
        get_image = os.path.join(source_folder,filename)
        move_image_to_cat = shutil.copy2(get_image,
                                        os.path.join(output_folder,c))
