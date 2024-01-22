from keras.datasets import cifar100
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import time
import matplotlib.pylab as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255

# plt.imshow(x_train[0], 'gray')
# plt.show()

# print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")
# x_train.shape=(50000, 32, 32, 3)
# x_test.shape=(10000, 32, 32, 3)
# y_train.shape=(50000, 1)
# y_test.shape=(10000, 1)

# print(np.unique(y_test,return_counts=True)) # 100종류, 전부 100개씩

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# model
model = Sequential()
model.add(Conv2D(32, (2,2), padding='same', input_shape=x_train.shape[1:], activation='relu'))
model.add(Conv2D(32, (2,2), padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (2,2), padding='same',activation='relu'))
model.add(Conv2D(32, (2,2), padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (2,2), padding='same',activation='relu'))
model.add(Conv2D(32, (2,2), padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (2,2), padding='same',activation='relu'))
model.add(Conv2D(32, (2,2), padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(100, activation='softmax'))

# model = Sequential()
# model.add(Conv2D(32, (2,2), padding='same', input_shape=x_train.shape[1:], activation='relu'))
# model.add(Conv2D(32, (2,2), padding='same',activation='relu'))
# model.add(Conv2D(32, (2,2), padding='same',activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(64, (2,2), padding='same',activation='relu'))
# model.add(Conv2D(64, (2,2), padding='same',activation='relu'))
# model.add(Conv2D(64, (2,2), padding='same',activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(100, activation='softmax'))

# compile & fit
start_time = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_acc', mode='max', patience=50,verbose=1)
hist = model.fit(x_train, y_train, epochs=1024, batch_size=2048, validation_data=(x_test,y_test), verbose=2, callbacks=es)
end_time = time.time()

# evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=0)

print(f"time: {end_time-start_time}sec")
print(f"LOSS: {loss[0]}\nACC:  {loss[1]}")

model.save(f"C:/_data/_save/keras31/CNN6_cifar100_ACC{loss[1]}.h5")

plt.title("CIFAR100")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.plot(hist.history['acc'],label='acc',color='red')
plt.plot(hist.history['val_acc'],label='val_acc',color='blue')
plt.legend()
plt.show()

# 0.4 넘기기

# time: 197.00242400169373sec
# LOSS: 2.7728023529052734
# ACC:  0.4724999964237213

# time: 280.78862738609314sec
# LOSS: 10.030982971191406
# ACC:  0.26170000433921814