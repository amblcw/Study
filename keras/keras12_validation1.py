#06_train1 data 카피
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import time
import math

#data
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,6])

x_val = np.array([6,7])
y_val = np.array([5,7])

x_test = np.array(range(8,11))
y_test = np.array(range(8,11))

#model generate
model = Sequential()
model.add(Dense(10,input_dim=1))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

#compile & fit
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=500,batch_size=1,verbose=2,validation_data=(x_val,y_val))

#evaluate & predict
loss = model.evaluate(x_test,y_test)
result = model.predict([7,11000000])
print(f"LOSS: {loss}\nRESULT: {result}")