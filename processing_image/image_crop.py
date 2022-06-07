import cv2
# import mediapipe as mp
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_face_mesh = mp.solutions.face_mesh

# 이미지 파일의 경우을 사용하세요.:
# IMAGE_FILES = ['./imageSet/training_set/Heart/heart (1).jpg']


# 얼굴부분 crop
# haarcascade 불러오기
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 이미지 불러오기
img = cv2.imread('./imageSet/training_set/Heart/heart (2).jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 얼굴 찾기
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cropped = img[y: y+h, x: x+w]
    resize = cv2.resize(cropped, (200, 200))
    # 이미지 저장하기
    cv2.imwrite("./imageSet/croptest.jpg", resize)

    cv2.imshow("crop&resize", resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()