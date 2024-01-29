#테스트폴더 쓰진말고 train폴더로
#변환시간도 체크하기

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
import os

# start_time = time.time()
path = "C:\\_data\\KAGGLE\\men_women\\"

BATCH_SIZE = int(1000)
IMAGE_SIZE = int(130)

load_path = path+f"datadata_{IMAGE_SIZE}px_"
x_train = np.load(load_path+"aug_x.npy")
y_train = np.load(load_path+"aug_y.npy")
x_test = np.load(load_path+"test_x.npy")
y_test = np.load(load_path+"test_y.npy")

hist = []

model = Sequential()
model.add(Conv2D(64,(3,3),padding='valid',strides=2,input_shape=x_train.shape[1:]))
model.add(MaxPooling2D())
model.add(Conv2D(64,(3,3),padding='valid',strides=2))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.25))
model.add(Conv2D(64,(2,2),padding='valid',activation='relu'))
model.add(Conv2D(64,(2,2),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.15))
model.add(Conv2D(64,(2,2),padding='same',activation='relu'))
model.add(Conv2D(64,(2,2),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.05))
model.add(Flatten())
model.add(Dense(2048,activation='relu'))
# model.add(Dense(512,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# compile & fit
s_time = time.time()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='val_loss',mode='auto',patience=50,restore_best_weights=True)
hist = model.fit(x_train,y_train,epochs=1024,batch_size=48,validation_split=0.2,verbose=2,callbacks=[es])
e_time = time.time()

# model = load_model("C:\_data\KAGGLE\cat-and-dog-classification-harper2022\model_save\\acc_0.852278.h5")

# evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=0)

print(f"LOSS: {loss[0]:.6f}\nACC:  {loss[1]:.6f}")
model.save(path+f"model_save\\acc_{loss[1]:.6f}.h5")



import matplotlib.pyplot as plt
if hist != []:
    plt.title("Men Women CNN")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    # plt.plot(hist.history['val_acc'],label='val_acc',color='red')
    # plt.plot(hist.history['acc'],label='acc',color='blue')
    plt.plot(hist.history['val_loss'],label='val_loss',color='red')
    plt.plot(hist.history['loss'],label='loss',color='blue')
    plt.legend()
    plt.show()

# fitting time: 391.9127sec
# LOSS: 0.410849
# ACC:  0.852278

# augment
# LOSS: 0.000000
# ACC:  1.000000