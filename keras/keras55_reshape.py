import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, Reshape, LSTM, Conv1D
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import time

# data
(x_train, y_train), (x_test,y_test) = mnist.load_data()


### 간단한 스케일링 방법 ###
x_train = np.asarray(x_train.reshape(60000,28,28,1)).astype(np.float32)/255
x_test = np.asarray(x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)).astype(np.float32)/255

# print(np.min(x_train),np.max(x_train))  #0.0 1.0

print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# model
model = Sequential()
model.add(Dense(9, input_shape=(28,28,1)))      # (28,28,1) ->  (28,28,9)
model.add(Conv2D(20, (2,2)))                    # (28,28,9) ->  (27,27,20)
model.add(Reshape(target_shape=(27,27,20)))     # (27,27,20)->  (27,27,20)
model.add(Conv2D(10, (2,2)))                    # (27,27,20)->  (26,26,10)
model.add(Reshape(target_shape=(26*26,10)))     # (26,26,10)->  (676,10)          
model.add(Conv1D(15,4))                         # (676,10)  ->  (673,15)
model.add(LSTM(10, return_sequences=True))      # (673,15)  ->  (673,10)
model.add(Flatten())                            # (673,10)  ->  (6730)
model.add(Dense(100, activation='relu'))        # (6730)    ->  (100)
model.add(Dropout(0.05))
model.add(Dense(units=10, activation='softmax'))# (100)     ->  (10)

model.summary()

start_time = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_acc', mode='auto', restore_best_weights=True)
hist = model.fit(x_train, y_train, batch_size=128, epochs=100, validation_split=0.2, verbose=2 )
end_time = time.time()
# evaluate & predict
loss = model.evaluate(x_test,y_test, verbose=0)
y_predict = model.predict(x_test, verbose=0)

print(f"time: {end_time - start_time}sec")
print(f"LOSS: {loss[0]}\nACC:  {loss[1]}")

# LOSS: 0.22471864521503448
# ACC:  0.983299970626831