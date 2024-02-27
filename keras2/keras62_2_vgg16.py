from keras.models import Sequential
from keras.layers import Dense, Flatten
import numpy as np
import tensorflow as tf
print(tf.__version__)   # 2.9.0
tf.random.set_seed(777) # 이쪽이 가중치 초기화에 영향
np.random.seed(777)

from keras.applications import VGG16
from keras.datasets import cifar10

# data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# model
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
vgg16.trainable = False # 가중치를 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.summary()
# Total params: 14,914,378
# Trainable params: 199,690
# Non-trainable params: 14,714,688
