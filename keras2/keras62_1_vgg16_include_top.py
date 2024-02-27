from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
print(tf.__version__)   # 2.9.0
tf.random.set_seed(777) # 이쪽이 가중치 초기화에 영향
np.random.seed(777)

from keras.applications import VGG16
from keras.datasets import cifar10

# model = VGG16()
# default params : include_top=True, input_shape=(224,224,3)
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
model = VGG16(
            #   weights='imagenet',
              include_top=False,    # fully-connected layer를 날린다, input_shape조정 가능해진다
              input_shape=(32,32,3),
              )
model.summary()

################### include_top = False ################### 
# 1. FC layer 날린다
# 2. input_shape 변경가능해진다




