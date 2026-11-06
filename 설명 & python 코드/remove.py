import os
import random

paths = ['/Users/gimgyuwon/Desktop/split_result/testing/10 19',
         '/Users/gimgyuwon/Desktop/split_result/testing/20 29',
         '/Users/gimgyuwon/Desktop/split_result/testing/30 39',
         '/Users/gimgyuwon/Desktop/split_result/validation/10 19',
         '/Users/gimgyuwon/Desktop/split_result/validation/20 29',
         '/Users/gimgyuwon/Desktop/split_result/validation/30 39']

target = 2571

random.seed(42)

for path in paths:
    img_names = [
        f for f in os.listdir(path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    n = len(img_names) - target

    if n<=0:
        print(path, " -> 이미", len(img_names), "장이라서 삭제X")
        continue
    remove_images = random.sample(img_names, n)

    for image in remove_images:
        os.remove(os.path.join(path, image))

    print(path)
    print("삭제전 : ", len(img_names))
    print("삭제 : ", n)
    print("남은 갯수 :", target)
    print()
    
