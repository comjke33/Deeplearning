'''
digit 숫자 인식 모델 제작
1. 먼저 dataset를 불러온다.
2. dataset를 분리시킨다. 
-> dataset은 train용, test용으로 분리하는데,
-> 이 비율은 7:3이 보편적으로 적당하다고 한다.
3. 이미지 전처리하는 신경망 모델을 제작 후 통과시킨다.
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

# tensorflow.keras. ~ 가 아니라,
# keras.~ 로 적어야 오류가 발생하지 않았음

# 1. dataset를 불러옴.
data_X = []     
data_y = []  
data_classes = len(os.listdir("digits/Digits"))
for i in range (0,data_classes):
    data_list = os.listdir("digits/Digits" +"/"+str(i))
    for j in data_list:
        pic = cv2.imread("digits/Digits" +"/"+str(i)+"/"+j)
        pic = cv2.resize(pic,(32,32))
        data_X.append(pic)
        data_y.append(i)
        if len(data_X) == len(data_y) :
            print("Total Dataponits = ",len(data_X))
# Labels and images
data_X = np.array(data_X)
data_y = np.array(data_y)

#Spliting the train validation and test sets
train_X, test_X, train_y, test_y = train_test_split(data_X,data_y,test_size=0.05)
train_X, valid_X, train_y, valid_y = train_test_split(train_X,train_y,test_size=0.2)
print("Training Set Shape = ",train_X.shape)
print("Validation Set Shape = ",valid_X.shape)
print("Test Set Shape = ",test_X.shape)

def Prep(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #making image grayscale
    img = cv2.equalizeHist(img) #Histogram equalization to enhance contrast
    img = img/255 #normalizing
    return img
train_X = np.array(list(map(Prep, train_X)))
test_X = np.array(list(map(Prep, test_X)))
valid_X= np.array(list(map(Prep, valid_X)))

#Reshaping the images
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2],1)
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], test_X.shape[2],1)
valid_X = valid_X.reshape(valid_X.shape[0], valid_X.shape[1], valid_X.shape[2],1)

#Augmentation
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
datagen.fit(train_X)

train_y = to_categorical(train_y, data_classes)
test_y = to_categorical(test_y, data_classes)
valid_y = to_categorical(valid_y, data_classes)

#Creating a Neural Network
model = Sequential()
model.add((Conv2D(60,(5,5),input_shape=(32, 32, 1) ,padding = 'Same' ,activation='relu')))
model.add((Conv2D(60, (5,5),padding="same",activation='relu')))
model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Dropout(0.25))
model.add((Conv2D(30, (3,3),padding="same", activation='relu')))
model.add((Conv2D(30, (3,3), padding="same", activation='relu')))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(500,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()

'''
This article was published as a part of the Data Science Blogathon

Introduction
Hello Readers!!
Deep Learning is used in many applications such as object detection, face detection, natural language processing tasks, and many more. In this blog I am going to build a model that will be used to solve unsolved Sudoku puzzles from an image using deep learning, We are going to libraries such as OpenCV and TensorFlow. If you want to know more about OpenCV, check this link. So let’s get started.

If you want to know about Python Libraries For Image Processing, then check this Link.
For more articles, click here
Sudoku Deep Learning image
Image Source 

The blog is divided into three parts:
Part 1: Digit Classification Model
We will be first building and training a neural network on the Char74k images dataset for digits. This model will help to classify the digits from the images.

Part 2: Reading and Detecting the Sudoku From an Image
This section contains, identifying the puzzle from an image with the help of OpenCV, classify the digits in the detected Sudoku puzzle using Part-1, finally getting the values of the cells from Sudoku and stored in an array.

Part3: Solving the Puzzle
We are going to store the array that we got in Pat-2 in the form of a matrix and finally run a recursion loop to solve the puzzle.

 

IMPORTING LIBRARIES 
Let’s import all the required libraries using the below commands:

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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
from PIL import Image
Part 1: Digit Classification Model 
In this section, we are going to use a digit classification model

LOADING DATA 
We are going to use an image dataset to classify the numbers in an image. Data is specified as features like images and labels as tags.

#Loading the data 
data = os.listdir("digits/Digits" )
data_X = []     
data_y = []  
data_classes = len(data)
for i in range (0,data_classes):
data_list = os.listdir("digits/Digits" +"/"+str(i))
    for j in data_list:
pic = cv2.imread("digits/Digits" +"/"+str(i)+"/"+j)
pic = cv2.resize(pic,(32,32))
data_X.append(pic)
data_y.append(i)
if len(data_X) == len(data_y) :
print("Total Dataponits = ",len(data_X))
# Labels and images
data_X = np.array(data_X)
data_y = np.array(data_y)
 

Sudoku Deep Learning datapoints
SPLITTING DATASET 
We are splitting the dataset into the train, test, and validation sets as we do in any machine learning problem.

#Spliting the train validation and test sets
train_X, test_X, train_y, test_y = train_test_split(data_X,data_y,test_size=0.05)
train_X, valid_X, train_y, valid_y = train_test_split(train_X,train_y,test_size=0.2)
print("Training Set Shape = ",train_X.shape)
print("Validation Set Shape = ",valid_X.shape)
print("Test Set Shape = ",test_X.shape)
 

Sudoku Deep Learning training set shape
Preprocessing the images for neural net 
In a preprocessing step, we preprocess the features (images) into grayscale, normalizing and enhancing them with histogram equalization. After that, convert them to NumPp arrays then reshaping them and data augmentation.

def Prep(img):
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #making image grayscale
img = cv2.equalizeHist(img) #Histogram equalization to enhance contrast
img = img/255 #normalizing
    return img
train_X = np.array(list(map(Prep, train_X)))
test_X = np.array(list(map(Prep, test_X)))
valid_X= np.array(list(map(Prep, valid_X)))
#Reshaping the images
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2],1)
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], test_X.shape[2],1)
valid_X = valid_X.reshape(valid_X.shape[0], valid_X.shape[1], valid_X.shape[2],1)
#Augmentation
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
datagen.fit(train_X)
 

One Hot Encoding

In this section, we are going to use one-hot encoding to labels the classes.

train_y = to_categorical(train_y, data_classes)
test_y = to_categorical(test_y, data_classes)
valid_y = to_categorical(valid_y, data_classes)
MODEL BUILDING

We are using a convolutional neural network for model building. It consists of the following steps:

#Creating a Neural Network
model = Sequential()
model.add((Conv2D(60,(5,5),input_shape=(32, 32, 1) ,padding = 'Same' ,activation='relu')))
model.add((Conv2D(60, (5,5),padding="same",activation='relu')))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.25))
model.add((Conv2D(30, (3,3),padding="same", activation='relu')))
model.add((Conv2D(30, (3,3), padding="same", activation='relu')))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(500,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()
model building Sudoku Deep Learning
In this step, we are going to compile the model and testing the model on the test set as shown below:

#Compiling the model
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon = 1e-08, decay=0.0)
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
#Fit the model
history = model.fit(datagen.flow(train_X, train_y, batch_size=32),
                              epochs = 30, validation_data = (valid_X, valid_y),
                              verbose = 2, steps_per_epoch= 200)

# Testing the model on the test set
score = model.evaluate(test_X, test_y, verbose=0)
print('Test Score = ',score[0])
print('Test Accuracy =', score[1])

# Randomly select an image from the dataset 
folder="sudoku-box-detection/aug"
a=random.choice(os.listdir(folder))
print(a)
sudoku_a = cv2.imread(folder+'/'+a)
plt.figure()
plt.imshow(sudoku_a)
plt.show()

#Preprocessing image to be read
sudoku_a = cv2.resize(sudoku_a, (450,450))
# function to greyscale, blur and change the receptive threshold of image
def preprocess(image):
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
blur = cv2.GaussianBlur(gray, (3,3),6) 
    #blur = cv2.bilateralFilter(gray,9,75,75)
threshold_img = cv2.adaptiveThreshold(blur,255,1,1,11,2)
    return threshold_img
threshold = preprocess(sudoku_a)
#let's look at what we have got
plt.figure()
plt.imshow(threshold)
plt.show()

# Finding the outline of the sudoku puzzle in the image
contour_1 = sudoku_a.copy()
contour_2 = sudoku_a.copy()
contour, hierarchy = cv2.findContours(threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(contour_1, contour,-1,(0,255,0),3)
#let's see what we got
plt.figure()
plt.imshow(contour_1)
plt.show()

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
def splitcells(img):
rows = np.vsplit(img,9)
boxes = []
    for r in rows:
cols = np.hsplit(r,9)
        for box in cols:
boxes.append(box)
    return boxes
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
'''
