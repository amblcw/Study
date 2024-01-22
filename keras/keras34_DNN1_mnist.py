import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import time

# data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")
# x_train.shape=(60000, 28, 28)
# x_test.shape=(10000, 28, 28)
# y_train.shape=(60000,)
# y_test.shape=(10000,)

# plt.imshow(x_train[0], 'gray')
# plt.show()
# print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64)
# print(pd.value_counts(y_test))
# 1    1135 # 2    1032 # 7    1028 # 3    1010 # 9    1009 # 4     982 # 0     980 # 8     974 # 6     958 # 5     892

# standard = StandardScaler().fit(x_train)
# x_train = standard.transform(x_train)
# x_test = standard.transform(x_test)

x_train = np.asarray(x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])).astype(np.float32)/255
x_test = np.asarray(x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])).astype(np.float32)/255
print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")


# y_train = y_train.reshape(-1,1)
# ohe = OneHotEncoder(sparse=False)
# y_train = ohe.fit_transform(y_train)
# print(type(y_train))



# model
model = Sequential()
model.add(Dense(2048, input_shape=(784,), activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()
'''
# Param = {(kenel_width * kenel_height) * channels + 1(bias 숫자)} * filters

 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 27, 27, 30)        150          {(2*2)*1 +1}*30 =150

 conv2d_1 (Conv2D)           (None, 26, 26, 20)        2420         {(2*2)*30 + 1}*20 = 2420

 flatten (Flatten)           (None, 13520)             0

 dense (Dense)               (None, 1000)              13521000

 dropout (Dropout)           (None, 1000)              0

 dense_1 (Dense)             (None, 10)                10010
=================================================================
Total params: 13,533,580
Trainable params: 13,533,580
Non-trainable params: 0
'''
# compile & fit
start_time = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_acc', mode='auto', restore_best_weights=True)
hist = model.fit(x_train, y_train, batch_size=128, epochs=100, validation_split=0.2, verbose=2 )
end_time = time.time()
# evaluate & predict
loss = model.evaluate(x_test,y_test, verbose=0)
y_predict = model.predict(x_test, verbose=0)

print(f"time: {end_time - start_time}sec")
print(f"LOSS: {loss[0]}\nACC:  {loss[1]}")

# LOSS: 0.22471864521503448
# ACC:  0.983299970626831

# padding
# time: 123.31915974617004sec
# LOSS: 0.29809698462486267
# ACC:  0.9793999791145325

# stride
# time: 62.33564519882202sec
# LOSS: 0.13542591035366058
# ACC:  0.9814000129699707

# MaxPooling2D
# time: 99.46496081352234sec
# LOSS: 0.1742657870054245
# ACC:  0.9850999712944031

# DNN
# time: 58.95517873764038sec
# LOSS: 0.24752727150917053
# ACC:  0.984499990940094