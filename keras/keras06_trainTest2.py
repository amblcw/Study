from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#data
x = np.array(range(1,11))   
y = np.array(range(1,11))

x_train = x[0:7]
y_train = y[0:7]

x_test = x[7:]
y_test = y[7:]

print(f"{x_train=}",f"{y_train=}",f"{x_test=}",f"{y_test=}",sep='\n')
# x_train=array([1, 2, 3, 4, 5, 6, 7])
# y_train=array([1, 2, 3, 4, 5, 6, 7])
# x_test=array([ 8,  9, 10])
# y_test=array([ 8,  9, 10])

#model generate
model = Sequential()
model.add(Dense(10,input_dim=1))
model.add(Dense(30))
model.add(Dense(100))
model.add(Dense(300))
model.add(Dense(100))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

#compile & fit
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=300,batch_size=1,verbose=2)

#evaluate & predict
loss = model.evaluate(x_test,y_test)
result = model.predict([7,11000000])
print(f"LOSS: {loss}\nRESULT: {result}")

# LOSS: 3.031649096259942e-13
# RESULT: [[7.0e+00]
#  [1.1e+07]]