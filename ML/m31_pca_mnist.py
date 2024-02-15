import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
import pandas as pd
import time

# data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")
# x_train.shape=(60000, 28, 28)
# x_test.shape=(10000, 28, 28)
# y_train.shape=(60000,)
# y_test.shape=(10000,)

# x = np.append(x_train,x_test, axis=0)
# x = np.concatenate([x_train,x_test], axis=0)
x = np.vstack([x_train,x_test])
print(x.shape)  # (70000, 28, 28)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

pca = PCA(n_components=x.shape[1])
x1 = pca.fit_transform(x)
EVR = pca.explained_variance_ratio_
EVR_sum = np.cumsum(EVR)
print(EVR_sum)
evr_sum = pd.Series(EVR_sum)
print(len(EVR_sum[EVR_sum >= 0.95]))
print(len(EVR_sum[EVR_sum >= 0.99]))
print(len(EVR_sum[EVR_sum >= 0.999]))
print(len(EVR_sum[EVR_sum >= 1.0]))

""" 
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

pca = PCA(n_components=x_train.shape[1]-1).fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

### 간단한 스케일링 방법 ###
x_train = np.asarray(x_train.reshape(60000,28,28,1)).astype(np.float32)/255
x_test = np.asarray(x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)).astype(np.float32)/255

# print(np.min(x_train),np.max(x_train))  #0.0 1.0

print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# model
model = Sequential()
model.add(Conv2D(filters=30, kernel_size=(2,2), input_shape=(28,28,1))) #Conv2D(filter:출력갯수,kernel_size=(2,2),input_shape=(28,28,1))
# model.add(Dropout(0.05))
model.add(Conv2D(20, (2,2)))
model.add(Conv2D(10, (2,2)))
model.add(Flatten())                                #일렬로 쭉 펴야마지막에 (batch_size, 10)을 맞춰줄수있다
model.add(Dense(1000, activation='relu'))           
model.add(Dropout(0.05))
# model.add(Dense(100, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# compile & fit
start_time = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_acc', mode='auto', restore_best_weights=True)
hist = model.fit(x_train, y_train, batch_size=128, epochs=100, validation_split=0.2, verbose=2 )
end_time = time.time()
# evaluate & predict
loss = model.evaluate(x_test,y_test, verbose=0)
y_predict = model.predict(x_test, verbose=0)

print(f"time: {end_time - start_time}sec")
print(f"LOSS: {loss[0]}\nACC:  {loss[1]}") """

# LOSS: 0.22471864521503448
# ACC:  0.983299970626831