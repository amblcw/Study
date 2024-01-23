from keras.datasets import fashion_mnist
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, BatchNormalization, MaxPooling2D, Conv2D, Flatten, Input
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import time
from sklearn.preprocessing import MinMaxScaler,StandardScaler

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# plt.imshow(x_train[0], 'gray')
# plt.show()

# print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")
# x_train.shape=(60000, 28, 28)
# x_test.shape=(10000, 28, 28)
# y_train.shape=(60000,)
# y_test.shape=(10000,)

# print(np.unique(y_test,return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000], dtype=int64))

xtr0, xtr1, xtr2, xtr3 = (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
xt0, xt1, xt2, xt3 = (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

x_train = x_train.reshape(xtr0, xtr1*xtr2*xtr3)
x_test = x_test.reshape(xt0, xt1*xt2*xt3)

# minmax = MinMaxScaler().fit(x_train)
# x_train = minmax.transform(x_train)
# x_test = minmax.transform(x_test)

standard = StandardScaler().fit(x_train)
x_train = standard.transform(x_train)
x_test = standard.transform(x_test)

x_train = x_train.reshape(xtr0, xtr1, xtr2, xtr3)
x_test = x_test.reshape(xt0, xt1, xt2, xt3)

# x_train = x_train.astype(np.float32) - 127.5
# x_test = x_test.astype(np.float32) - 127.5
# x_train /= 127.5
# x_test /= 127.5

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# model
# model = Sequential()
# model.add(Conv2D(32, (2,2), input_shape=x_train.shape[1:], activation='relu'))
# model.add(Conv2D(32, (2,2), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(32, (2,2), padding='same',activation='relu'))
# model.add(Conv2D(32, (2,2), padding='same',activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(10, activation='softmax'))

input = Input(shape=x_train.shape[1:])
c1 = Conv2D(32,(2,2),activation='relu')(input)
c2 = Conv2D(32,(2,2),activation='relu')(c1)
b1 = BatchNormalization()(c2)
m1 = MaxPooling2D()(b1)
dr1 = Dropout(0.25)(m1)
c3 = Conv2D(32,(2,2),activation='relu')(dr1)
c4 = Conv2D(32,(2,2),activation='relu')(c3)
b2 = BatchNormalization()(c4)
m2 = MaxPooling2D()(b2)
dr2 = Dropout(0.25)(m2)
f1 = Flatten()(dr2)
d1 = Dense(1024,activation='relu')(f1)
dr3 = Dropout(0.1)(d1)
output = Dense(10, activation='softmax')(dr3)

model = Model(inputs=input,outputs=output)

# compile & fit
start_time = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='auto', patience=100, verbose=1)
hist = model.fit(x_train, y_train, epochs=1024, batch_size=2048, validation_data=(x_test,y_test), verbose=2, callbacks=[es])
end_time = time.time()

# evaluate
loss = model.evaluate(x_test,y_test, verbose=0)

print(f"time: {end_time-start_time}sec")
print(f"LOSS: {loss[0]}\nACC:  {loss[1]}")

# time: 766.1504282951355sec
# LOSS: 0.17683808505535126
# ACC:  0.9399999976158142

# few padding
# time: 193.0108208656311sec
# LOSS: 0.22429831326007843
# ACC:  0.9265000224113464

# function 
# time: 108.75242805480957sec
# LOSS: 0.3682504892349243
# ACC:  0.9283999800682068

# scaled -1 to 1
# time: 391.8226647377014sec
# LOSS: 1.8357616662979126
# ACC:  0.7400000095367432

# scaled by MinMaxScaler
# time: 111.51893448829651sec
# LOSS: 0.37955421209335327
# ACC:  0.9291999936103821

# scaled by StandardScaler
# time: 117.99499416351318sec
# LOSS: 0.3999277353286743
# ACC:  0.9200999736785889