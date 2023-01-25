import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True)
base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

# TEST
print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

BATCH_SIZE = 100
IMG_SHAPE  = 150

# -----------------------------------------------
# 여기까지는 모두 같다.

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)

# 이미지를 수평 방향으로 반전시킨다.
# 이미지의 인식 정도를 높이기 위해, 먼저 이미지를 수평 방향으로 뒤집는 과정을 진행한다.
train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE,directory=train_dir,shuffle=True,target_size=(IMG_SHAPE,IMG_SHAPE))

# TEST
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

# 이미지를 회전시킨다
image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)

train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE,directory=train_dir,shuffle=True,target_size=(IMG_SHAPE, IMG_SHAPE))

# TEST
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

# 이미지를 확대시킨다.
image_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5)

train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE,directory=train_dir,shuffle=True,target_size=(IMG_SHAPE, IMG_SHAPE))

# TEST
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

# 모든 회전, 반전, 줌을 합친다.
# 이와 같은 과정을 통해 모델이 훈련 데이터를 기억하기 더 어렵게 하고 유효성 검사 dataset에서 더 잘 수행되도록 한다.
image_gen_train = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE, directory=train_dir,shuffle=True,target_size=(IMG_SHAPE,IMG_SHAPE),class_mode='binary')

# TEST
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

# Validation Data 생성기를 제작한다.
# 유효성 검사의 dataset은 증강 없이 원본 이미지를 유지한다.
image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE,directory=validation_dir,target_size=(IMG_SHAPE, IMG_SHAPE),class_mode='binary')

# 모델 생성
# 모델 dropout이 추가된 것 외에는 달라진 것이 없다.
# 이는 모델 자체의 탄력성을 높인다고 한다.
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2)
])

# 모델 컴파일
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# 모델 요약문 출력
model.summary()

# 모델 훈련
# 전에는 에포크를 100으로 했을 때 정확도가 매우 낮았는데, 증강 이미지를 도입한 후 88%까지 증강하는 것을 볼 수 있다.
epochs=100
history = model.fit_generator(train_data_gen,steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)

# 모델 결과 출력
# 이 코드를 실행하면 에포크의 변화에 따라 유효성 검사 수치를 보여주는데, 우리는 이 그래프에서 에포크가 60일때 가장 정확도가 높다는 것을 알 수 있기 때문에 에포크 수치를 100이 아닌 60으로 맞춰야한다는 것을 보여준다.
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
