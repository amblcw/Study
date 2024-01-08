from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#data
# x = np.array(range(1,11))   #주가, 금리, 환율
# y = np.array(np.arange(1,11))

x_train = np.array(range(1,8))
y_train = np.array([1,2,3,4,6,5,7])

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
model.fit(x_train,y_train,epochs=500,batch_size=1,
          verbose=100) #verbose= 0:slient, 1:all display(default), 2:one-line display, 3+:Epoch only

#evaluate & predict
loss = model.evaluate(x_test,y_test)
result = model.predict([7,11000000])
print(f"LOSS: {loss}\nRESULT: {result}")

# LOSS: 0.0006753153284080327
# RESULT: [[7.0500989e+00]
#  [1.0854442e+07]]
# 