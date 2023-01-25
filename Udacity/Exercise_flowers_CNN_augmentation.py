# 필요한 라이브러리 
import os
import numpy as np
import glob
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt

# 필요한 ImageDataGenerator 라이브러리 로드
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# URL에서 dataset를 불러옴
_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

# dataset의 매개변수는 각각 document를 살펴봐야함.
zip_file = tf.keras.utils.get_file(origin=_URL,
                                   fname="flower_photos.tgz",
                                   extract=True)

# base_dir 변수 선언
base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

# 꽃의 종류 class 선언
classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

# class에 맞게 확장자 잘라내고 directory 정렬
for cl in classes:
  img_path = os.path.join(base_dir, cl)
  images = glob.glob(img_path + '/*.jpg')
  print("{}: {} Images".format(cl, len(images)))
  train, val = images[:round(len(images)*0.8)], images[round(len(images)*0.8):]

  for t in train:
    if not os.path.exists(os.path.join(base_dir, 'train', cl)):
      os.makedirs(os.path.join(base_dir, 'train', cl))
    shutil.move(t, os.path.join(base_dir, 'train', cl))

  for v in val:
    if not os.path.exists(os.path.join(base_dir, 'val', cl)):
      os.makedirs(os.path.join(base_dir, 'val', cl))
    shutil.move(v, os.path.join(base_dir, 'val', cl))
    
# train directory와 validation directory 다르게 선언
# cats_and_dogs dataset과 다른 점은 validation이 val이라는 이름으로 들어가있다는 점.
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# 훈련에 필요한 total_dir를 계산하기 위해 각각 꽃 종류의 directory를 모두 선언
train_roses_dir = os.path.join(train_dir, 'roses')
train_daisy_dir = os.path.join(train_dir, 'daisy')
train_dandelion_dir = os.path.join(train_dir, 'dandelion')
train_sunflowers_dir = os.path.join(train_dir, 'sunflowers')
train_tulips_dir = os.path.join(train_dir, 'tulips')

# 훈련에 필요한 validation directory 꽃의 종류별로 각각 선언
validation_roses_dir = os.path.join(val_dir, 'roses')
validation_daisy_dir = os.path.join(val_dir, 'daisy')
validation_dandelion_dir = os.path.join(val_dir, 'dandelion')
validation_sunflowers_dir = os.path.join(val_dir, 'sunflowers')
validation_tulips_dir = os.path.join(val_dir, 'tulips')

# 꽃 종류마다 train 횟수 계산을 위해 변수 선언
# len는 length의 약자
num_roses_tr = len(os.listdir(train_roses_dir))
num_daisy_tr = len(os.listdir(train_daisy_dir))
num_dandelion_tr = len(os.listdir(train_dandelion_dir))
num_sunflowers_tr = len(os.listdir(train_sunflowers_dir))
num_tulips_tr = len(os.listdir(train_tulips_dir))

# 각 꽃의 종류마다 validation를 계산하기 위해 변수 선언
num_roses_val = len(os.listdir(validation_roses_dir))
num_daisy_val = len(os.listdir(validation_daisy_dir))
num_dandelion_val = len(os.listdir(validation_dandelion_dir))
num_sunflowers_val = len(os.listdir(validation_sunflowers_dir))
num_tulips_val = len(os.listdir(validation_tulips_dir))

# batch_size와 이미지 크기 선언
batch_size = 100
IMG_SHAPE = 150

# ImageDataGenerator 함수로 이미지 가져오고 수평 방향으로 반전
image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
train_data_gen = image_gen.flow_from_directory(batch_size = batch_size, directory= train_dir, shuffle = True, target_size = (IMG_SHAPE, IMG_SHAPE))

# TEST
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

# TEST
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

# ImageDataGenerator 함수로 이미지 가져오고 회전
image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)
train_data_gen = image_gen.flow_from_directory(batch_size = batch_size, directory = train_dir, shuffle=True, target_size = (IMG_SHAPE, IMG_SHAPE))

# TEST
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

# ImageDataGenerator 함수로 이미지 가져오고 줌
image_gen = ImageDataGenerator(rescale= 1./255, zoom_range=0.5)
train_data_gen = image_gen.flow_from_directory(batch_size= batch_size, directory= train_dir, shuffle = True, target_size= (IMG_SHAPE, IMG_SHAPE))

# TEST
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

# ImageDataGenerator 함수로 이미지 반전, 회전, 줌 모두 실행
image_gen_train = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size, directory=train_dir,shuffle=True,target_size=(IMG_SHAPE,IMG_SHAPE),class_mode='binary')

# TEST
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

# 이미지 유효성 검사 set 제작
# 255로 나누어 0 ~ 1 사이의 값으로 만든다
image_gen_val =  ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,directory=val_dir,target_size=(IMG_SHAPE, IMG_SHAPE),class_mode='sparse')

# 모델 제작
# 지시 사항대로 첫번째는 16개 layer, 두번째는 32개 layer, 세번째는 64개 layer로 만들었다.
# Dropout를 20%로 설정해서 overfitting을 막았다.
# Flatten 함수를 추가하라고 해서 했다.
# 512 dense로 설정했다.
# 마지막으로 출력층은 꽃 종류가 5개 이므로 class 갯수를 5로 했고, 활성화함수는 softmax로 설정하라길래 햇다.
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation = 'softmax')
])

# 모델을 컴파일하는 과정
# optimizer는 adam으로 했고, 손실함수도 아래와 같이 설정했다.
# metrics를 accuracy로 설정하라고 해서 했다.
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# 에포크는 80로 설정했다.
# 그래서 훈련시킬 때 진짜 엄청 많이 걸렸다.
# 왜냐면 30*80 = 2400
epochs = 80

# 모델을 훈련했다.
# 기존 코드에서 total_train과 total_val에 관련된 부분만 수정해서 완성했다.
total_train = num_roses_tr + num_daisy_tr + num_dandelion_tr + num_sunflowers_tr + num_tulips_tr
total_val = num_roses_val + num_daisy_val + num_dandelion_val + num_sunflowers_val + num_tulips_val
history = model.fit_generator(train_data_gen,steps_per_epoch=int(np.ceil(total_train / float(batch_size))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(batch_size)))
)

# 모델 훈련 후 training과 validation과 관련된 그래프를 plot했다.




