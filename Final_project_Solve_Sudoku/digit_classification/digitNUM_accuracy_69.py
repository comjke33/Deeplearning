'''
digit 숫자 인식 모델 제작
1. 먼저 dataset를 불러온다.
2. dataset를 분리시킨다. 
-> dataset은 train용, test용으로 분리하는데,
-> 이 비율은 7:3이 보편적으로 적당하다고 한다.
3. 이미지 전처리 후, 신경망 모델을 제작 후 통과시킨다.
4. 스도쿠 이미지를 하나 가지고 온다.
5. 명암을 감지해서 가장 큰 사각형을 인식한 후, 새로운 변수에 저장함.
6. 셀을 9로 나누어서 총 81개의 부분으로 나눈 뒤, 안에 들어있는 숫자를 모델에 보내서 파악함.
7. 81개의 숫자를 9*9 matrix에 저장한다.
8. 이중 for문으로 접근해서 오류가 있는 스도쿠 사진인지 감지한다.

(선택)
9-1. 만약 스도쿠 이미지가 덜 풀려져있다면? -> 빈 공간은 0으로 처리 후
9-2. 스도쿠 퍼즐을 풀이하는 모듈을 짜서 사용한다.
9-3. 만약 가능하다면, 빈공간에 포지션을 잡아서 자동으로 숫자를 합성해 정답 이미지를 제작해 보여준다.
9-4. 정답 이미지를 다운로드할 수 있으면 좋을 것 같다.
9-5. 판별하는데 걸린 시간이나 스도쿠가 풀리기 전 푼 갯수에 비례한 퍼센트지를 보여줘도 좋을 것 같다.
9-6. 걸린 시간과 퍼센트지에 맞는 평가 메세지를 text로 띄우는 것도 좋을 것 같다.
'''

# 필요한 라이브러리를 불러옴
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os, random
import cv2
from glob import glob
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K
from keras.preprocessing import image
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
from PIL import Image


epochNUM = 44

# 1. dataset를 불러옴.
data_classes=len(os.listdir("E:/sudoku/digit"))
print(data_classes)
data_X = []
data_Y = []
for i in range(0,data_classes):
    data_list = os.listdir("E:/sudoku/digit"+"/"+str(i))
    for j in data_list:
        try:
            pic = cv2.imread("E:/sudoku/digit" +"/"+str(i)+"/"+str(j))
            pic = cv2.resize(pic,(32,32)) 
            data_X.append(pic)
            data_Y.append(i)
            # 성공했으면, 문구 내보내기
            if len(data_X)==len(data_Y):
                print("Total Datapoints= ", len(data_X))
        except:
            print("예외가 발생했습니다.")

# X, Y에 넣는다.
# 배열화해서 접근이 용이하게 한다.
data_X = np.array(data_X)
data_Y = np.array(data_Y)

# 2. dataset 분리
# dataset를 train 데이터와 test 데이터 그리고 Validation 데이터로 나눈다.
# 먼저 Test 데이터로 5%의 데이터, Train 데이터로 95%의 데이터를 떼어주었다.
# 그리고 Train 데이터에서 다시 Train 데이터 80% 그리고 validation 데이터로 20%를 나누어 주었다.
train_X, test_X, train_Y, test_Y = train_test_split(data_X, data_Y, test_size=0.05)
train_X, validation_X, train_Y, validation_Y = train_test_split(train_X, train_Y, test_size=0.2)
print("Training Set Shape = ",train_X.shape)
print("Validation Set Shape = ",validation_X.shape)
print("Test Set Shape = ",test_X.shape)

# 3. 이미지 전처리
# RGB -> GRAYSCALE: 색깔을 없앤다.
# equalizeHist(): 히스토그램 평활화 방법을 통해 픽셀값을 조정해 선명한 이미지를 얻는다.
# 0 ~ 255의 명암도를 0 ~ 1 사이의 실수값으로 변환한다.

def Prep(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    img = cv2.equalizeHist(img) 
    img = img/255 #normalizing
    return img

# prep 함수를 거쳐온 train, test, validation X 데이터값을 맵핑한 후, 
# 리스트에 넣고 배열에 저장한다.
train_X = np.array(list(map(Prep, train_X)))
test_X = np.array(list(map(Prep, test_X)))
validation_X= np.array(list(map(Prep, validation_X)))

# X 0,1,2 즉 3차원을 1차원으로 바꿔준다.
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2],1)
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], test_X.shape[2],1)
validation_X = validation_X.reshape(validation_X.shape[0], validation_X.shape[1], validation_X.shape[2],1)

# Augmentation 이용 
# 반전, 회전, 줌
datagen = ImageDataGenerator(horizontal_flip=True)
datagen.fit(train_X)

datagen = ImageDataGenerator(zoom_range=0.5)
datagen.fit(train_X)

datagen = ImageDataGenerator(rotation_range=45)
datagen.fit(train_X)

datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
datagen.fit(train_X)

datagen = ImageDataGenerator(
      rotation_range=70,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
datagen.fit(train_X)

datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
datagen.fit(train_X)

# One Hot Encoding 이용
# One-Hot encoding이란: 값을 0과 1로 인식하는 과정
# 클래스에 맞춰 라벨을 붙이기 위한 것
train_Y = to_categorical(train_Y, data_classes)
test_Y = to_categorical(test_Y, data_classes)
validation_Y = to_categorical(validation_Y, data_classes)

# 모델 제작
# 입력층
model = Sequential()
model.add((Conv2D(60,(5,5),input_shape=(32, 32, 1) ,padding = 'Same' ,activation='relu')))
 
#은닉층1
model.add((Conv2D(60, (5,5),padding="same",activation='relu')))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add((Dropout(0.25)))

#은닉층2
model.add((Conv2D(30, (3,3),padding="same", activation='relu')))
model.add((Conv2D(30, (3,3), padding="same", activation='relu')))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())

#출력층
model.add(Dense(500,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# 모델 정보 출력
model.summary()

# 모델 컴파일 
# RMSprop 옵티마이저
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon = 1e-08, decay=0.0)
model.compile(optimizer = optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# 모델 학습 시작!
history = model.fit(datagen.flow(train_X, train_Y, batch_size=32),
                              epochs = epochNUM, validation_data = (validation_X, validation_Y),
                              verbose = 2, steps_per_epoch= 40)
                      
                      
# Test 데이터를 이용해 중간 점검
score = model.evaluate(test_X, test_Y, verbose=0)
print('Test Score = ',score[0])
print('Test Accuracy =', score[1])

epochs_range = range(epochNUM)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
 
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
'''

# 스도쿠 데이터셋에서 랜덤으로 이미지를 하나 뽑아옴.
# 전처리 전 스도쿠 원본 이미지를 창 띄워 보여줌.
# folder="E:/sudoku/sudoku"
# a=random.choice(os.listdir(folder))
# print(a)
sudoku_a= cv2.imread("E:/sudoku/sudoku/2.jpg")
# sudoku_a = cv2.imread(folder+'/'+a)
plt.figure()
plt.imshow(sudoku_a)
plt.show()


# 이미지 전처리
sudoku_a = cv2.resize(sudoku_a, (450,450))
def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray, (3,3),6) 
    #blur = cv2.bilateralFilter(gray,9,75,75)
    threshold_img = cv2.adaptiveThreshold(blur,255,1,1,11,2)
    return threshold_img
threshold = preprocess(sudoku_a)

# TEST
plt.figure()
plt.imshow(threshold)
plt.show()


# 선정한 이미지에서 테두리 모양을 찾아냄.
# opencv의 findContours를 이용해서 외곽선 정보를 검출한다.
# cv2.drawContours(이미지, 외곽선 검출 모드, 외곽선 근사화 방법, 검출된 외곽선 좌표, 외곽선 계층 정보, 좌표 값 이동 오프셋)
# 이 함수를 이용해서 외곽선을 검출해서 변수에 리스트로 저장한다.
contour_1 = sudoku_a.copy()
contour_2 = sudoku_a.copy()
contour, hierarchy = cv2.findContours(threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# drawContours 함수를 이용해서 이미지와 검출된 contour값 
# 그리고 외곽선 인덱스를 음수인 -1로 지정해서 모든 외곽선을 그렸다.
# RGB값으로 외곽선 색상을 지정하고 thickness를 3으로 
# 최대 외곽선 레벨을 3으로 설정했다.
cv2.drawContours(contour_1, contour,-1,(0,255,0),3)

# 현재 contour_1라는 변수에 외곽선을 딴 이미지가 저장되어있다.
plt.figure()
plt.imshow(contour_1)
plt.show()

# 외곽선 부분 리스트를 가져와서 
# 적당한 값인 50을 넘은 부분만을 arclength 함수로 외곽선 길이를 구한다.
# 그리고 approxPolyDP 함수를 이용해서 외곽선을 단순화시킵니다.
def main_outline(contour):
    biggest = np.array([])
    max_area = 0
    for i in contour:
        area = cv2.contourArea(i)
        if area >50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i , 0.02* peri, True)
            if area > max_area and len(approx) ==4:
                biggest = approx
                max_area = area
    return biggest ,max_area

# 이미지의 외곽선 네 꼭짓점의 위치를 points라는 리스트에 저장한다.
def reframe(points):
    points = points.reshape((4, 2))
    points_new = np.zeros((4,1,2),dtype = np.int32)
    add = points.sum(1)
    points_new[0] = points[np.argmin(add)]
    points_new[3] = points[np.argmax(add)]
    diff = np.diff(points, axis =1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]
    return points_new

# 네 꼭짓점이 구해진 사각형을 기존 이미지에서 잘라내 따로 저장해 반환한다.
def splitcells(img):
    rows = np.vsplit(img,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes

# 외곽 단순화 함수를 거치고 외곽선이 가장 큰 부분을 이미지 처리한다.
black_img = np.zeros((450,450,3), np.uint8)
biggest, maxArea = main_outline(contour)
if biggest.size != 0:
    biggest = reframe(biggest)
    cv2.drawContours(contour_2,biggest,-1, (0,255,0),10)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[450,0],[0,450],[450,450]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imagewrap = cv2.warpPerspective(sudoku_a,matrix,(450,450))
    imagewrap =cv2.cvtColor(imagewrap, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(imagewrap)
plt.show()

sudoku_cell = splitcells(imagewrap)
#Let's have alook at the last cell
plt.figure()
plt.imshow(sudoku_cell[58])
plt.show()

def CropCell(cells):
    Cells_croped = []
    for image in cells:
        img = np.array(image)
        img = img[4:46, 6:46]
        img = Image.fromarray(img)
        Cells_croped.append(img)
    return Cells_croped
sudoku_cell_croped= CropCell(sudoku_cell)
#Let's have alook at the last cell
plt.figure()
plt.imshow(sudoku_cell_croped[58])
plt.show()

# Reshaping the grid to a 9x9 matrix
grid = np.reshape(sudoku_cell_croped,(9,9))
grid
'''
