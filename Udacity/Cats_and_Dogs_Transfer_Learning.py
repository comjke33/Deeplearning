# 기존에 제작된 모델을 가져다가 출력층만 다른 모델로 재학습하는 전이학습 과정을 거친다.
# 그래서 새로운 결과를 도출할 수 있는 모델을 제작하는 것이다.
# 현재 기존 모델인 MobileNet -> Cats and Dogs 모델로 전이 학습(Transfer Learning)

# Tensorflow와 필요한 라이브러리 모두 로드
import tensorflow as tf
import matplotlib.pylab as plt

import tensorflow_hub as hub
import tensorflow_datasets as tfds

from tensorflow.keras import layers

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# URL를 통해 MobileNet 모델 로드
CLASSIFIER_URL ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
IMAGE_RES = 224

# MobileNet은 입력 해상도가 224*224인 컬러 이미지를 입력으로 받는다.
# 해상도를 래핑하는 단일 케라스 레이어인 MobileNet이 있다.
model = tf.keras.Sequential([
    hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
])

# CNN 신경망을 위해 필요한 numpy 수학 함수 로드
import numpy as np
import PIL.Image as Image

# TEST
grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize((IMAGE_RES, IMAGE_RES))
grace_hopper 

grace_hopper = np.array(grace_hopper)/255.0
grace_hopper.shape

# 예측 클래스에 단일 이미지 호출
result = model.predict(grace_hopper[np.newaxis, ...])
result.shape

# 1001개의 클래스가 있는 ImageNet이라는 dataset에서 훈련된 MobileNet이기 때문에,
# 지금 이 이미지를 1001개의 클래스 중 하나로 판별한다
# 이 명령어를 실행하면 1001개 중 몇 번째 클래스인지 나온다.
predicted_class = np.argmax(result[0], axis=-1)
predicted_class

# 몇 번째 클래스인지 나왔다면, 그것이 무엇을 의미하는 것인지 라벨을 출력하기 위해 필요한 명령어
# 군복이라고 나온다.
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title())

#cats vs dogs 모델을 로드한다
(train_examples, validation_examples), info = tfds.load(
    'cats_vs_dogs', 
    with_info=True, 
    as_supervised=True, 
    split=['train[:80%]', 'train[80%:]'],
)

num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes

# 모델의 해상도 크기를 출력하기 위해 적은 명령어
# 이 모델은 전의 모델과 다르다. 더 작다.
for i, example_image in enumerate(train_examples.take(3)):
  print("Image {} shape: {}".format(i+1, example_image[0].shape))

# 0 ~ 1의 값으로 나타내기 위한 것. 
def format_image(image, label):
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return image, label

BATCH_SIZE = 32

# prefetch는 최적화를 위해 적어준 것일 뿐 중요한 것은 아님
train_batches      = train_examples.shuffle(num_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)

# 이미지 테스트
image_batch, label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

result_batch = model.predict(image_batch)

predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]
predicted_class_names

plt.figure(figsize=(10,9))
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.subplots_adjust(hspace = 0.3)
  plt.imshow(image_batch[n])
  plt.title(predicted_class_names[n])
  plt.axis('off')
_ = plt.suptitle("ImageNet predictions")

# MobileNet에서 최종 계층만 제외한 모델을 가져온다
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
feature_extractor = hub.KerasLayer(URL,
                                   input_shape=(IMAGE_RES, IMAGE_RES,3))

# MobileNet의 출력층과 Cats vs Dogs의 입력층을 연결하기 위해 batch를 32에 맞춰둔 것이다
feature_batch = feature_extractor(image_batch)
print(feature_batch.shape)

# MobileNet의 모델의 일부를 수정 불가하게 하였다
# 그래야지 출력층에 내가 추가하고자 하는 모델을 가져올 수 있다
feature_extractor.trainable = False

# 새로운 분류 layer를 추가한다
model = tf.keras.Sequential([
  feature_extractor,
  layers.Dense(2)
])

model.summary()

# 모델을 컴파일한다
# 컴파일한다는 것은, 모델을 훈련시키 전 설정을 해주는 과정이다
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

# 에포크 6으로 model.fit() 훈련시켰다
EPOCHS = 6
history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)


# 그래프 출력
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

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

# 예측된 batch값을 변수로 선언한다
class_names = np.array(info.features['label'].names)
class_names

predicted_batch = model.predict(image_batch)
predicted_batch = tf.squeeze(predicted_batch).numpy()
predicted_ids = np.argmax(predicted_batch, axis=-1)
predicted_class_names = class_names[predicted_ids]
predicted_class_names

print("Labels: ", label_batch)
print("Predicted labels: ", predicted_ids)

plt.figure(figsize=(10,9))
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.subplots_adjust(hspace = 0.3)
  plt.imshow(image_batch[n])
  color = "blue" if predicted_ids[n] == label_batch[n] else "red"
  plt.title(predicted_class_names[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")




