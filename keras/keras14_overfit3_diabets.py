from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import pandas as pd

#data
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,random_state=333)

#model
model = Sequential()
model.add(Dense(16,input_dim=10,activation='relu'))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#compile & fit
model.compile(loss='mse',optimizer='adam')
hist = model.fit(x_train,y_train,epochs=128,batch_size=16,validation_split=0.3,verbose=2)

#evaluate & predict
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)

print(f"{loss=}\n{r2=}")

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
plt.title("당뇨병")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(hist.history['loss'],c='red',label='loss',marker='.')
plt.plot(hist.history['val_loss'],c='green',label='val_loss',marker='.')
plt.legend()
plt.grid()
plt.show()