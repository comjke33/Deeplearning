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
                              epochs = 10, validation_data = (validation_X, validation_Y),
                              verbose = 2, steps_per_epoch= 40)
                      
                      
# Test 데이터를 이용해 중간 점검
score = model.evaluate(test_X, test_Y, verbose=0)
print('Test Score = ',score[0])
print('Test Accuracy =', score[1])
