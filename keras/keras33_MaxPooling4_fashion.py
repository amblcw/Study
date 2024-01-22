from keras.datasets import fashion_mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, MaxPooling2D, Conv2D, Flatten
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import time

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

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)

x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# model
model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=x_train.shape[1:], activation='relu'))
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# model.add(Conv2D(32, (2,2),activation='relu'))
# model.add(Conv2D(32, (2,2),activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
model.add(Conv2D(32, (2,2), padding='same',activation='relu'))
model.add(Conv2D(32, (2,2), padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))

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