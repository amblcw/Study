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
x = np.load(load_path+"x.npy")
y = np.load(load_path+"y.npy")

print(x.shape,y.shape)

r = int(np.random.uniform(1,1000))
r = 965
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=r, stratify=y)
hist = []

photo_gen = ImageDataGenerator(
    rescale=1./255
)

photo_data = photo_gen.flow_from_directory(
    "C:\_data\KAGGLE\men_women\photo\\",
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE,        #batch_size 너무 크게주면 에러나옴
    class_mode='binary',
    shuffle=False
)

photo = photo_data[0][0]

# model = Sequential()
# model.add(Conv2D(64,(3,3),padding='valid',strides=2,input_shape=x_train.shape[1:]))
# model.add(MaxPooling2D())
# model.add(Conv2D(64,(3,3),padding='valid',strides=2))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())
# model.add(Dropout(0.25))
# model.add(Conv2D(64,(2,2),padding='valid',activation='relu'))
# model.add(Conv2D(64,(2,2),padding='valid',activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())
# model.add(Dropout(0.15))
# model.add(Conv2D(64,(2,2),padding='same',activation='relu'))
# model.add(Conv2D(64,(2,2),padding='same',activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())
# model.add(Dropout(0.05))
# model.add(Flatten())
# model.add(Dense(4096,activation='relu'))
# model.add(Dense(512,activation='relu'))
# model.add(Dense(1,activation='sigmoid'))

# # compile & fit
# s_time = time.time()
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
# es = EarlyStopping(monitor='val_loss',mode='auto',patience=100,restore_best_weights=True)
# hist = model.fit(x_train,y_train,epochs=1024,batch_size=48,validation_data=(x_test,y_test),verbose=2,callbacks=[es])
# e_time = time.time()

model = load_model("C:\_data\KAGGLE\men_women\model_save\\acc_0.779456.h5")

# evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=0)

y_predict = np.around(model.predict(photo))

if y_predict[0][0] == 0:
    print("man")
else:
    print("woman")
# print(f"LOSS: {loss[0]:.6f}\nACC:  {loss[1]:.6f}")
# model.save(path+f"model_save\\acc_{loss[1]:.6f}.h5")



import matplotlib.pyplot as plt
if hist != []:
    plt.title("Men Women CNN")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.plot(hist.history['val_acc'],label='val_acc',color='red')
    plt.plot(hist.history['acc'],label='acc',color='blue')
    # plt.plot(hist.history['val_loss'],label='val_loss',color='red')
    # plt.plot(hist.history['loss'],label='loss',color='blue')
    plt.legend()
    plt.show()

# fitting time: 391.9127sec
# LOSS: 0.410849
# ACC:  0.852278