import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import time

# data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.asarray(x_train.reshape(60000,28,28,1)).astype(np.float32)/255
x_test = np.asarray(x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)).astype(np.float32)/255
print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# model
model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same', input_shape=(28,28,1))) #Conv2D(filter:출력갯수,kernel_size=(2,2),input_shape=(28,28,1))
model.add(Conv2D(10, (2,2), padding='same'))
model.add(Conv2D(10, (2,2), padding='same'))
model.add(GlobalAveragePooling2D())
# model.add(Flatten())                              
model.add(Dense(1000, activation='relu'))           
model.add(Dropout(0.05))
model.add(Dense(units=10, activation='softmax'))

model.summary()
'''
Flatten
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

GlobalAveragePooling
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 30)        150

 conv2d_1 (Conv2D)           (None, 28, 28, 20)        2420

 conv2d_2 (Conv2D)           (None, 28, 28, 10)        810

 global_average_pooling2d (G  (None, 10)               0
 lobalAveragePooling2D)

 dense (Dense)               (None, 1000)              11000

 dropout (Dropout)           (None, 1000)              0

 dense_1 (Dense)             (None, 10)                10010

=================================================================
Total params: 24,390
Trainable params: 24,390
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