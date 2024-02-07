from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import sklearn.preprocessing

from sklearn.svm import LinearSVC

#data
datasets = load_iris()

x = datasets.data
y = datasets.target


# print(x)
print(f"{x.shape=}, {y.shape=}")        #x.shape=(150, 4), y.shape=(150,)
# print(np.unique(y, return_counts=True)) #array([0, 1, 2]), array([50, 50, 50])

r = int(np.random.uniform(1,1000))
r=326
x_train , x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=r,stratify=y)

#model
# model = Sequential()
# model.add(Dense(128,input_dim=4,activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8,  activation='relu'))
# model.add(Dense(3,  activation='softmax'))
model = LinearSVC(C=110)

#compile & fit
# model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
# model.fit(x_train,y_train,epochs=100,batch_size=1,validation_split=0.4,verbose=2)
model.fit(x_train,y_train)

#evaluate & predict
# loss = model.evaluate(x_test,y_test)
loss = model.score(x_test,y_test)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)

#결과값 출력
print(f"{r=}\nACC: {loss}, {acc}")



# Epoch 312: early stopping
# 1/1 [==============================] - 0s 62ms/step
# r=167
# LOSS: 0.05837009474635124
# ACC:  1.0

#stratify 적용 후

# patience = 30
# Epoch 43: early stopping
# 1/1 [==============================] - 0s 52ms/step
# r=326
# LOSS: 0.0691366046667099
# ACC:  1.0(1.0by loss[1])
# y_test's    contents: 0=10, 1=10, 2=10
# y_predict's contents: 0=10, 1=10, 2=10

# SVM
# r=326
# ACC: 1.0