from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

# 분류 대상 카테고리 선택하기 --- (※1)
# caltech_dir = "./image/101_ObjectCategories"
categories = [ 'Heart', 'Oblong', 'Oval', 'Round', 'Square' ]
nb_classes = len(categories)

# 이미지 크기 지정 --- (※2)
image_w = 64
image_h = 64
pixels = image_w * image_h * 3

# 이미지 데이터 읽어 들이기 --- (※3)
X = []
Y = []
for idx, cat in enumerate(categories):
    # 레이블 지정 --- (※4)
    # One Hot Encoding (비트를 이용하여 labeling)
    label = [0 for i in range(nb_classes)]
    label[idx] = 1

# 이미지 --- (※5)
    IMAGE_DIR = './imageSet/processed_training_cropped_image/'+cat+'/'
    files = glob.glob(IMAGE_DIR+"/*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f) # --- (※6)
        print("idx: ", str(idx), str(i)+' '+str(img.size))
        # img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        X.append(data)
        Y.append(label)
        # if i % 10 == 0:
        #     print(i, "\n", data)
X = np.array(X)
print(X.shape)
Y = np.array(Y)
print(Y.shape)
# 학습 전용 데이터와 테스트 전용 데이터 구분 --- (※7)
X_train, X_test, y_train, y_test = train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./imageSet/cropped_face_dataset.npy", xy)
# print("ok,", len(Y))