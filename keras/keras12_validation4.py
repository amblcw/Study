from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

#data
x = np.arange(1,17)
y = np.arange(1,17)

r = int(np.random.uniform(1,1000))
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=13/16, random_state=333)
# x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size=10/13,shuffle=False)

# print(x_train,x_val,x_test)
# print(y_train,y_val,y_test)

#model
model = Sequential()
model.add(Dense(10,input_dim=1,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1))

#compile & fit
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=1024,batch_size=1,validation_split=3/13,verbose=2)

#evaluate & predict
loss = model.evaluate(x_test,y_test)
result = model.predict([7,11000000])
print(f"{r=}\nLOSS: {loss}\nRESULT: {result}")
