from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.array([1,2,3])
y = np.array([1,2,3])

model = Sequential()
model.add(Dense(5, input_shape=(1,)))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
'''
 Layer (type)                Output Shape              Param #
=================================================================       
 dense (Dense)               (None, 5)                 10

 dense_1 (Dense)             (None, 4)                 24

 dense_2 (Dense)             (None, 2)                 10

 dense_3 (Dense)             (None, 1)                 3

각 레이어에 bias가 추가되기에 계산 결과가 이렇게 나온다
'''