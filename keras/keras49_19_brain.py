#테스트폴더 쓰진말고 train폴더로
#변환시간도 체크하기

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import time

start_time = time.time()
path = "C:\\_data\\etc\\brain\\"
train_path = path+"train\\"
test_path = path+"test\\"

BATCH_SIZE = int(200)
IMAGE_SIZE = int(200)

np_path = "C:\\_data\\_save_npy\\"
x = np.load(np_path+f"keras39_1_x_train.npy")
y = np.load(np_path+f"keras39_1_y_train.npy")

x = x.reshape(x.shape[0],x.shape[1],x.shape[2]*x.shape[3]).astype(np.float32) / 255
y = y.reshape(y.shape[0],y.shape[1],y.shape[2]*y.shape[3]).astype(np.float32) / 255
# test = test.reshape(test.shape[0],test.shape[1],test.shape[2]*test.shape[3]).astype(np.float32) / 255

r = int(np.random.uniform(1,1000))
r = 965
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=r, stratify=y)

model = Sequential()
model.add(Conv2D(32,(3,3),padding='valid',strides=2,input_shape=x_train.shape[1:]))
model.add(MaxPooling2D())
model.add(Conv2D(32,(3,3),padding='valid',strides=2))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.15))
model.add(Conv2D(32,(3,3),padding='valid',activation='relu'))
model.add(Conv2D(32,(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.15))
model.add(Conv2D(64,(2,2),padding='valid',activation='relu'))
model.add(Conv2D(64,(2,2),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.15))
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# compile & fit
s_time = time.time()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='val_loss',mode='auto',patience=50,restore_best_weights=True)
hist = model.fit(x_train,y_train,epochs=1024,batch_size=48,validation_data=(x_test,y_test),verbose=2,callbacks=[es])
e_time = time.time()

# evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=0)

print(f"fitting time: {e_time-s_time:.4f}sec")
print(f"LOSS: {loss[0]:.6f}\nACC:  {loss[1]:.6f}")
model.save(path+f"model_save\\acc_{loss[1]:.6f}.h5")

import matplotlib.pyplot as plt

if hist != []:
    plt.title("Brain CNN")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.plot(hist.history['val_acc'],label='val_acc',color='red')
    plt.plot(hist.history['acc'],label='acc',color='blue')
    # plt.plot(hist.history['val_loss'],label='val_loss',color='red')
    # plt.plot(hist.history['loss'],label='loss',color='blue')
    plt.legend()
    plt.show()

# augment
# LOSS: 0.000000
# ACC:  1.000000
