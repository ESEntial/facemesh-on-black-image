import cv2
import mediapipe as mp
import os
import glob
import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## 함수
# 디렉토리가 존재하지 않으면 생성
def makedirs(path): 
   try: 
        os.makedirs(path) 
   except OSError: 
       if not os.path.isdir(path): 
           raise


# SAVE_DIR = './result'
labels = [ 'Heart', 'Oblong', 'Oval', 'Round', 'Square' ]
setTypes = [ 'testing', 'training' ]


# mesh용
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
# detection용
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils



point_idx = 0
# For static images:
for setType in setTypes:
  for label in labels:
    SAVE_DIR = './imageSet/mediapipe_result/'+setType+'/'+label+'/'
    makedirs(SAVE_DIR)
    IMAGE_DIR = './imageSet/processed_'+setType+'_cropped_image/'+label+'/'
    print(IMAGE_DIR)
    # processed_training_cropped_image
    IMAGE_FILES = glob.glob(IMAGE_DIR+'*.jpg')
    print(IMAGE_FILES)
          
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
        # mesh 기본 설정
        static_image_mode=True, # False => 입력 이미지를 비디오로
        max_num_faces=2,        # 감지할 최대 얼굴 수
        refine_landmarks=True,  # 랜드마크 세부화
        min_detection_confidence=0.5) as face_mesh: # 얼굴 탐지 모델의 신뢰 값
      for idx, file in enumerate(IMAGE_FILES):
        currentFileName = os.path.basename(file)
        
        image = cv2.imread(file)
        # Convert the BGR image to RGB before processing.
        convert_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mesh_results = face_mesh.process(convert_image)

        # Print and draw face mesh landmarks on the image.
        if not mesh_results.multi_face_landmarks:
          continue
        
        # mesh를 그릴 이미지 복사
        annotated_image = image.copy()

        for face_landmarks in mesh_results.multi_face_landmarks:
          mp_drawing.draw_landmarks(
              image=annotated_image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_TESSELATION,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_tesselation_style())
          mp_drawing.draw_landmarks(
              image=annotated_image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_CONTOURS,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_contours_style())
          mp_drawing.draw_landmarks(
              image=annotated_image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_IRISES,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_iris_connections_style())
        
        ## landmark 정규화 값 시각화
        print(mesh_results.multi_face_landmarks[0].landmark)
        pos_x = list()
        pos_y = list()
        pos_z = list()
        for value in mesh_results.multi_face_landmarks[0].landmark:
          pos_x.append(value.x)
          pos_y.append(value.y)
          pos_z.append(value.z)
        
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pos_x, pos_y, pos_z, c='r', marker='o', s=3)
        plt.scatter(pos_x, pos_y, pos_z)
        
        # plt.show()  # 이미지 띄우기
        plt.savefig(SAVE_DIR+currentFileName) # 이미지 저장
        