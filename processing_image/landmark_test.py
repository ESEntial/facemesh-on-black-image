import glob
import os

import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

## 함수
# 디렉토리가 존재하지 않으면 생성
def makedirs(path): 
   try: 
        os.makedirs(path) 
   except OSError: 
       if not os.path.isdir(path): 
           raise


# 경로 설정 변수
readErrorImageList = []
saveErrorImageList = []
labels = [ 'Heart', 'Oblong', 'Oval', 'Round', 'Square' ]
setTypes = [ 'testing', 'training' ]

# 얼굴부분 crop
# haarcascade 불러오기
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Main
for setType in setTypes:
    print("Processing "+setType)
    for label in labels:
        print("Processing "+label)
        IMAGE_DIR = './imageSet/'+setType+'_set/'+label+'/'
        SAVE_DIR ='./imageSet/processed_'+setType+'_set/'+label+'/'
        makedirs(IMAGE_DIR)
        makedirs(SAVE_DIR)
        IMAGE_FILES = glob.glob(IMAGE_DIR+'*.jpg')
        
        # 표현되는 랜드마크의 굵기와 반경
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=0)
        # mean = 0
        # oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
        #         400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
        #         54, 103, 67, 109]
        # cheek_left = [123, 50, 36, 137, 205, 206, 177, 147, 187, 207, 213, 216, 215, 192, 138,
        #         214, 212, 135]
        # cheek_right = [266, 280, 352, 366, 425, 426, 411, 427, 376, 401, 436, 433, 435, 416,
        #         434, 367, 364, 432]

        # face_whole = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
        #         400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
        #         54, 103, 67, 109, 123, 50, 36, 137, 205, 206, 177, 147, 187, 207, 213, 216, 215, 192, 138,
        #         214, 212, 135, 266, 280, 352, 366, 425, 426, 411, 427, 376, 401, 436, 433, 435, 416,
        #         434, 367, 364, 432]

        # x_list = np.linspace(0, 0, len(face_whole))
        # y_list = np.linspace(0, 0, len(face_whole))
        # z_list = np.linspace(0, 0, len(face_whole))


        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
            for idx, file in enumerate(IMAGE_FILES):
                currentFileName = os.path.basename(file)


                # 이미지 불러오기
                image = cv2.imread(file)
                if image is None:
                    print("READ ERROR!!" + currentFileName)
                    readErrorImageList.append(currentFileName)
                    continue
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # 얼굴 crop
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cropped = image[y: round((y+h)*1.2), x: round((x+w)*1.2)]
                    resize = cv2.resize(cropped, (512, 512), cv2.INTER_LANCZOS4)
                image = resize
                
                # 작업 전에 BGR 이미지를 RGB로 변환합니다.
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # 이미지에 출력하고 그 위에 얼굴 그물망 경계점을 그립니다.
                if not results.multi_face_landmarks:
                    continue
                #annotated_image = image.copy()
                annotated_image = np.zeros((512, 512, 3), np.uint8)
                
                
                ih, iw, ic = annotated_image.shape
                for face_landmarks in results.multi_face_landmarks:
                    # 각 랜드마크를 image에 overlay 시켜줌
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec
                    )

                writeReturn = cv2.imwrite(SAVE_DIR+ currentFileName, annotated_image)
                if writeReturn == True:
                    print("successfully saved image!!" + currentFileName)
                else: 
                    print("SAVE ERROR!!" + currentFileName)
                    saveErrorImageList.append(currentFileName)
                
        print("Done "+label)
    print("Done "+setType)
    
for idx, readErrorImage in enumerate(readErrorImageList):
    print(idx, "Read Error Image: ", readErrorImage)
    
for idx, saveErrorImage in enumerate(saveErrorImageList):
    print(idx, "Save Error Image: ", saveErrorImage)
    

