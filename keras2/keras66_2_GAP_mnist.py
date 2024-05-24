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

import tensorflow as tf
import random
RANDOM_STATE = random.randint(1,10000)
RANDOM_STATE = 6184
tf.random.set_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

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
model.add(MaxPooling2D())
model.add(Conv2D(20, (2,2), padding='same'))
model.add(Conv2D(20, (2,2), padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(40, (2,2), padding='same'))
model.add(Conv2D(40, (2,2), padding='same'))
model.add(MaxPooling2D())
# model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(1000, activation='relu'))           
model.add(Dropout(0.05))
model.add(Dense(units=10, activation='softmax'))

model.summary()

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
print(f"Random state: ",RANDOM_STATE)

# GlobalAveragePooling2D
# time: 260.6653392314911sec
# LOSS: 0.06826115399599075
# ACC:  0.988099992275238
# Random state:  6184

# time: 263.8555371761322sec
# LOSS: 0.08318240940570831
# ACC:  0.9922999739646912
# Random state:  6184