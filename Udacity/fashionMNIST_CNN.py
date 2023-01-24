import tensorflow as tf

# tensorflow datasets를 import함.
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# 라이브러리 불러옴.
#numpy는 인공지능 연산 과정에서 필요한 수학 함수
#matplotlib.pyplot은 그래프 그릴때 사용하는 라이브러리
import math
import numpy as np
import matplotlib.pyplot as plt

# 에러 발생과 관련된 로깅 라이브러리와 관련 명령어
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Fashion mnist dataset를 불러오는 명령어
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# class를 나눈다. -> 옷 종류
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal','Shirt',   'Sneaker',  'Bag',   'Ankle boot']

# TEST
# metadata 명령어를 사용해서 train과 test 예시 각각을 출력
# 중요한 것 아님
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))

# 이미지는 images, labels로 분류.
# 이미지 값은 흑백이라서 0~255로 나와있음
# 그런데 이미지가 모델에 입력되려면 0~1 사이의 값으로 정의되있어야함.
# 그래서 255로 나눠주는 작업이 필요함. -> 소수점으로 들어가겠지!
# 이 함수는 모델에서 이미지를 가져와서(tf.cast 명령어 이용) images와 labels를 리턴하는 함수임.
def normalize(images, labels):
  images = tf.cast(images, tf.float32)
  images /= 255
  return images, labels

# map함수를 이용해서 dataset의 각각 요소를 훈련을 위해 적용시키는 과정.
# 데이터셋은 train용과 test용 두가지로 분류되어있다.
train_dataset =  train_dataset.map(normalize)
test_dataset  =  test_dataset.map(normalize)

# dataset이 처음 사용되면, 이미지를 디스크(저장소)로부터 로드되어야함.
# 그래서 데이터셋 캐쉬를 메모리에 저장하고 이는 훈련을 더 빠르게 가속화시켜줌.
# 만약, 내가 모델을 on-device 형식으로 제작한다면 꼭 필요한 과정이다. 로컬 디스크 내에 모델이 저장될 수 있도록 해야함.
train_dataset =  train_dataset.cache()
test_dataset  =  test_dataset.cache()

# 하나의 이미지를 가져와서 색 차원을 없앤다.
# 단순히 흑백 이미지만 이용하기 때문 -> 모양만 판단하면 되는 인식 모델이므로
for image, label in test_dataset.take(1):
  break
image = image.numpy().reshape((28,28))

# TEST
# 이미지 하나를 가져와서 plot한다. 
# 중요한 것 아님
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

# TEST
# 25개의 이미지를 가져와서 plot한다.
# 중요한 것 아님
plt.figure(figsize=(10,10))
i = 0
for (image, label) in test_dataset.take(25):
    image = image.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
    i += 1
plt.show()


# layer를 만든다.
# convolution을 위해 32개의 출력값, 3*3 kernel, padding을 설정했다.
# maxpooling은 2*2
# 이 과정을 64로 한번 더 반복한다.
# 출력층 tf.keras.layers.dense -> relu, softmax 활성화함수를 이용해서 출력층을 만든다. 잘보면 relu는 128층, softmax는 10층이다.
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# 모델을 컴파일한다.
# 손실 함수를 최소화하기 위한 모델의 내부 파라미터를 조절하는 것을 optimizer라고 함.
# loss는 손실함수. 낮으면 낮을수록 좋은 것.
# metrics는 training과 testing을 어떻게 진행할 것인지 정함. 정확하게 하겠다~ 라는 accuracy 이용.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

# 모델을 훈련한다.
# epoch값만큼 데이터를 반복해 읽으면서 훈련한다. dataset.repeat() (epochs라는 파라미터가 존재함)
# dataset.shuffle() 데이터셋을 랜덤으로 섞어서 배운다.
# dataset.batch() 모델 파라미터를 업데이트할 때 몇개의 sample마다 가중치를 갱신할 것인지 설정함. batch_size라는 파라미터가 존재함.
BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

# 훈련
model.fit(train_dataset, epochs=10, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))
