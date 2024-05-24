from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import time
import matplotlib.pyplot as plt

import tensorflow as tf
import random
RANDOM_STATE = random.randint(1,10000)
RANDOM_STATE = 8422
tf.random.set_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# model
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(2,2),padding='same', strides=2, input_shape=x_train.shape[1:],activation='relu'))
model.add(Conv2D(32,(2,2),padding='same',activation='relu'))
model.add(Conv2D(32,(2,2),padding='same',activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.05))
model.add(Conv2D(filters=64,kernel_size=(2,2),padding='same', strides=2,activation='relu'))
model.add(Conv2D(64,(2,2),padding='same',activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.05))
model.add(Conv2D(filters=128,kernel_size=(2,2),padding='same', strides=2,activation='relu'))
model.add(Conv2D(128,(2,2),padding='same',activation='relu'))
model.add(GlobalAveragePooling2D())
# model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10,activation='softmax'))
model.summary()

# compile & fit
start_time = time.time()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='val_acc',mode='auto',patience=50,restore_best_weights=True)
hist = model.fit(x_train,y_train,epochs=1000,batch_size=1024,validation_data=(x_test,y_test),verbose=2,callbacks=[es])
end_time = time.time()

# model = load_model("주소")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             

# evaluate & predict
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

print(f"time: {end_time-start_time}sec")
print(f"LOSS: {loss[0]}\nACC:  {loss[1]}")
print(f"Random state:",RANDOM_STATE)

model.save(f"C:/_data/_save/keras31/CNN5_cifar10_ACC{loss[1]}.h5")

plt.title("CIFAR10")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.plot(hist.history['acc'],label='acc',color='red')
plt.plot(hist.history['val_acc'],label='val_acc',color='blue')
plt.legend()
plt.show()

# GlobalAveragePooling2D
# time: 118.85527658462524sec
# LOSS: 1.019281029701233
# ACC:  0.7114999890327454
# Random state: 8422

# Flatten
# time: 106.01268529891968sec
# LOSS: 0.9328374862670898
# ACC:  0.7081999778747559
# Random state: 8422