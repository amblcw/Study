from keras.datasets import cifar100
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import time
import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import random
RANDOM_STATE = random.randint(1,10000)
RANDOM_STATE = 8422
tf.random.set_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# model
model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=x_train.shape[1:], activation='relu'))
model.add(Conv2D(32, (2,2),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (2,2),activation='relu'))
model.add(Conv2D(32, (2,2),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (2,2),activation='relu'))
model.add(Conv2D(32, (2,2),activation='relu'))
model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='softmax'))

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
print(f"RANDOM_STATE: ",RANDOM_STATE)

model.save(f"C:/_data/_save/keras31/CNN6_cifar100_ACC{loss[1]}.h5")

plt.title("CIFAR100")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.plot(hist.history['acc'],label='acc',color='red')
plt.plot(hist.history['val_acc'],label='val_acc',color='blue')
plt.legend()
plt.show()

# Flatten
# time: 157.22122621536255sec
# LOSS: 2.526672601699829
# ACC:  0.44940000772476196
# RANDOM_STATE:  8422

# GlobalAveragePooling2D
# time: 235.95450115203857sec
# LOSS: 2.0663905143737793
# ACC:  0.47029998898506165
# RANDOM_STATE:  8422