from keras.datasets import cifar10
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization,Input
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")
# x_train.shape=(50000, 32, 32, 3)
# x_test.shape=(10000, 32, 32, 3)
# y_train.shape=(50000, 1)
# y_test.shape=(10000, 1)
# print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],dtype=int64))

# acc > 0.77

xtr0, xtr1, xtr2, xtr3 = (x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3])
xt0, xt1, xt2, xt3 = (x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])

x_train = x_train.reshape(xtr0, xtr1*xtr2*xtr3)
x_test = x_test.reshape(xt0, xt1*xt2*xt3)

minmax = MinMaxScaler().fit(x_train)
x_train = minmax.transform(x_train)
x_test = minmax.transform(x_test)

# standard = StandardScaler().fit(x_train)
# x_train = standard.transform(x_train)
# x_test = standard.transform(x_test)

x_train = x_train.reshape(xtr0, xtr1, xtr2, xtr3)
x_test = x_test.reshape(xt0, xt1, xt2, xt3)

# x_train = x_train.astype(np.float32) - 127.5
# x_test = x_test.astype(np.float32) - 127.5
# x_train /= 127.5
# x_test /= 127.5


aug_data_gen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    shear_range=30,
    zoom_range=0.2,
    fill_mode='nearest',
)

aug_data = aug_data_gen.flow(
    x_train,
    y_train,
    batch_size=x_train.shape[0],
    shuffle=False
)

aug_x, aug_y = aug_data.next()

print(f"{aug_x.shape=} {aug_y.shape=}")

x_train = np.concatenate([x_train,aug_x])
y_train = np.concatenate([y_train,aug_y])

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")


input = Input(shape=x_train.shape[1:])
c1 = Conv2D(32,(2,2),padding='same',strides=2,activation='swish')(input)
c2 = Conv2D(32,(2,2),padding='same',activation='swish')(c1)
c3 = Conv2D(32,(2,2),padding='same',activation='swish')(c2)
m1 = MaxPooling2D()(c3)
dr1 = Dropout(0.1)(m1)
c4 = Conv2D(64,(2,2),padding='same',activation='swish')(dr1)
c5 = Conv2D(64,(2,2),padding='same',activation='swish')(c4)
c6 = Conv2D(64,(2,2),padding='same',activation='swish')(c5)
m2 = MaxPooling2D()(c6)
dr2 = Dropout(0.1)(m2)
fl = Flatten()(dr2)
d1 = Dense(512,activation='swish')(fl)
dr3 = Dropout(0.1)(d1)
output = Dense(10,activation='softmax')(dr3)

model = Model(inputs=input,outputs=output)

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

model.save(f"C:/_data/_save/keras31/CNN5_cifar10_ACC{loss[1]}.h5")

plt.title("CIFAR10")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.plot(hist.history['acc'],label='acc',color='red')
plt.plot(hist.history['val_acc'],label='val_acc',color='blue')
plt.legend()
plt.show()

# time: 714.9462764263153sec
# LOSS: 1.3513137102127075
# ACC:  0.7827000021934509

# scale by StandardScaler
# time: 177.7754030227661sec
# LOSS: 1.577500343322754
# ACC:  0.723800003528595

# augment
# time: 209.8251211643219sec
# LOSS: 1.3657841682434082
# ACC:  0.6901000142097473