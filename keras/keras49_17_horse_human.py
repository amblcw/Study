#테스트폴더 쓰진말고 train폴더로
#변환시간도 체크하기

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization, LSTM
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
import os

# start_time = time.time()
path = "C:\\_data\\image\\horse-or-human\\"

BATCH_SIZE = int(500)
IMAGE_SIZE = int(300)

load_path = path+f"data_{IMAGE_SIZE}px"
x = np.load(load_path+"_aug_x.npy")
y = np.load(load_path+"_aug_y.npy")

x = x.reshape(x.shape[0],x.shape[1],x.shape[2]*x.shape[3]).astype(np.float32) / 255
# y = y.reshape(y.shape[0],y.shape[1],y.shape[2]*y.shape[3]).astype(np.float32) / 255
# test = test.reshape(test.shape[0],test.shape[1],test.shape[2]*test.shape[3]).astype(np.float32) / 255

print("x, y shape: ",x.shape,y.shape)  #(1027, 300, 300, 3) (1027, 2) onehot이 되어있음

r = int(np.random.uniform(1,1000))
# r = 965
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=r, stratify=y)
hist = []

# model
model = Sequential()
# model.add(Conv2D(32,(3,3),padding='valid',strides=2,input_shape=x_train.shape[1:]))
# model.add(MaxPooling2D())
# model.add(Conv2D(32,(3,3),padding='valid',strides=2))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())
# model.add(Dropout(0.15))
# model.add(Conv2D(32,(2,2),padding='valid',activation='relu'))
# model.add(Conv2D(32,(2,2),padding='valid',activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())
# model.add(Dropout(0.15))
# model.add(Conv2D(64,(2,2),padding='same',activation='relu'))
# model.add(Conv2D(64,(2,2),padding='same',activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())
# model.add(Dropout(0.15))
# model.add(Flatten())
model.add(LSTM(32,input_shape=x_train.shape[1:]))
model.add(Dense(2048,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# compile & fit
s_time = time.time()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='val_acc',mode='auto',patience=30,restore_best_weights=True)
hist = model.fit(x_train,y_train,epochs=1024,batch_size=16,validation_split=0.2,verbose=2,callbacks=[es])
e_time = time.time()


# evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=0)


print(f"LOSS: {loss[0]:.6f}\nACC:  {loss[1]:.6f}")
model.save(path+f"model_save\\acc_{loss[1]:.6f}.h5")


import matplotlib.pyplot as plt
if hist != []:
    plt.title("Horse&Human CNN")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.plot(hist.history['val_acc'],label='val_acc',color='red')
    plt.plot(hist.history['acc'],label='acc',color='blue')
    # plt.plot(hist.history['val_loss'],label='val_loss',color='red')
    # plt.plot(hist.history['loss'],label='loss',color='blue')
    plt.legend()
    plt.show()

# LOSS: 0.001914
# ACC:  1.000000

# augment
# LOSS: 0.006363
# ACC:  0.996350