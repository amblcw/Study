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
x = np.arange(1,17)
y = np.arange(1,17)

x_train = x[:10]
y_train = y[:10]

x_val = x[10:13]
y_val = y[10:13]

x_test = x[13:]
y_test = y[13:]

print(x_train,x_val,x_test,sep='\n')

#model generate
model = Sequential()
model.add(Dense(10,input_dim=1,activation='relu'))
# model.add(Dense(8,activation='relu'))
# model.add(Dense(10,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1))

#compile & fit
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=1024,batch_size=1,verbose=2,validation_data=(x_val,y_val))

#evaluate & predict
loss = model.evaluate(x_test,y_test)
result = model.predict([7,11000000])
print(f"LOSS: {loss}\nRESULT: {result}")

#461 layers epo=512, all relu
# LOSS: 1.8189894035458565e-12
# RESULT: [[6.9999995e+00]
#  [1.0999998e+07]]

#421 layers, epo=512, all relu
# LOSS: 1.515824466814808e-12
# RESULT: [[7.0000005e+00]
#  [1.1000000e+07]]

#441 layers, epo=512, all relu
# LOSS: 6.063298192519884e-13
# RESULT: [[6.9999995e+00]
#  [1.0999999e+07]]

#441 layers, epo=2048, all relu, 1300부근에서 업데이트 안됨
# LOSS: 9.094947017729282e-13
# RESULT: [[7.0000000e+00]
#  [1.0983617e+07]]

#위와 동일, epo=1024
# LOSS: 3.3348139787808817e-12
# RESULT: [[6.9999995e+00]
#  [1.0999999e+07]]

#위와 동일
# LOSS: 6.063298192519884e-13
# RESULT: [[7.0e+00]
#  [1.1e+07]]

#위와 동일
# LOSS: 1.8189894035458565e-12
# RESULT: [[7.0000000e+00]
#  [1.0999999e+07]]

#10 4 1 layers, epo=1024, batch=1, all relu
# LOSS: 0.0
# RESULT: [[7.0000005e+00]
#  [1.1000000e+07]]