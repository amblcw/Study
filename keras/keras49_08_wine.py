from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

datasets = load_wine()
x = datasets.data
y = datasets.target

# print(x.shape,y.shape)  #(178, 13) (178,)
# print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48

y = to_categorical(y)
print(y,y.shape,sep='\n') #(178,3)

r = int(np.random.uniform(1,1000))
r = 398
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=r,stratify=y)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler().fit(x_train)    #최솟값을 0 최댓값을 1로 스케일링
# scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
scaler = RobustScaler().fit(x_train)    #

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
# test_csv = test_csv.reshape(test_csv.shape[0],test_csv.shape[1],1)

#model
model = Sequential()
model.add(LSTM(128,input_shape=(13,1)))
# model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

#compile & fit
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='val_acc',mode='max',patience=50,restore_best_weights=True)
hist = model.fit(x_train,y_train,epochs=4096,batch_size=1,validation_split=0.2,verbose=2,callbacks=[es])

#evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=0)
y_predict = np.argmax(model.predict(x_test,verbose=0),axis=1)
y_test = np.argmax(y_test,axis=1)

print(f"{r=} \nLOSS: {loss[0]} \nACC:  {accuracy_score(y_test,y_predict)}({loss[1]} by loss[1])")

plt.figure(figsize=(12,9))
plt.title("Wine Classification")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.plot(hist.history['acc'],label='acc',color='red')
plt.plot(hist.history['val_acc'],label='val_acc',color='blue')
plt.legend()
# plt.show()

# r=398
# LOSS: 0.13753800094127655
# ACC:  1.0(1.0 by loss[1])

# MinMaxScaler
# LOSS: 0.11889973282814026
# ACC:  0.9444444444444444(0.9444444179534912 by loss[1])

# StandardScaler
# LOSS: 0.09690777212381363
# ACC:  0.9444444444444444(0.9444444179534912 by loss[1])

# MaxAbsScaler
# LOSS: 0.30003201961517334
# ACC:  0.9722222222222222(0.9722222089767456 by loss[1])

# RobustScaler
# LOSS: 0.15712764859199524
# ACC:  0.9722222222222222(0.9722222089767456 by loss[1])

# RNN 
# LOSS: 0.10301997512578964
# ACC:  0.9722222222222222(0.9722222089767456 by loss[1])