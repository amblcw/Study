from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#data
x = np.array(range(1,11))   #주가, 금리, 환율
print(x)
print(x.shape)

y = np.array([1,2,3,4,6,5,7,8,9,10])
print(y)
print(y.shape)

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
model.fit(x,y,epochs=180,batch_size=1,verbose=2)

#evaluate & predict
loss = model.evaluate(x,y)
result = model.predict([7,11000000])
print(f"LOSS: {loss}\nRESULT: {result}")

# LOSS: 0.30893880128860474
# RESULT: [[6.6104574e+00]
#  [1.0139330e+07]]