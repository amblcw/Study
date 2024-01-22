import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

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

x_train = np.asarray(x_train.reshape(60000,28,28,1)).astype(np.float32)/255
x_test = np.asarray(x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)).astype(np.float32)/255
print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# y_train = y_train.reshape(-1,1)
# ohe = OneHotEncoder(sparse=False)
# y_train = ohe.fit_transform(y_train)
# print(type(y_train))



# model
model = Sequential()
model.add(Conv2D(filters=30, kernel_size=(2,2), padding='same', strides=2,input_shape=(27,27,1))) #Conv2D(filter:출력갯수,kernel_size=(2,2),input_shape=(28,28,1))
model.add(Conv2D(20, (2,2)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())                                #일렬로 쭉 펴야마지막에 (batch_size, 10)을 맞춰줄수있다
model.add(Dense(1000, activation='relu'))           
model.add(Dropout(0.05))
# model.add(Dense(100, activation='relu'))
model.add(Dense(units=10, activation='softmax'))    
# Dense input_shape = (batch_size, input_dim),              output_shape = (batch_size, units)
# Conv2D input_shape = (batch_size, rows, colums, channels),output_shape = (channels, new_rows, new_colums, filters)

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
''' stride = 2 (stride 끝에 남는 자투리는 버린다)
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 14, 14, 30)        150

 conv2d_1 (Conv2D)           (None, 13, 13, 20)        2420

 flatten (Flatten)           (None, 3380)              0

 dense (Dense)               (None, 1000)              3381000

 dropout (Dropout)           (None, 1000)              0

 dense_1 (Dense)             (None, 10)                10010

=================================================================
Total params: 3393580 (12.95 MB)
Trainable params: 3393580 (12.95 MB)
Non-trainable params: 0 (0.00 Byte)
'''
# compile & fit
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# es = EarlyStopping(monitor='val_acc', mode='auto', restore_best_weights=True)
# hist = model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=2, )

# # evaluate & predict
# loss = model.evaluate(x_test,y_test, verbose=0)
# y_predict = model.predict(x_test, verbose=0)

# print(f"LOSS: {loss[0]}\nACC:  {loss[1]}")

# LOSS: 0.4081042408943176
# ACC:  0.9818000197410583