from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#data
x = np.array(range(10))   #주가, 금리, 환율
print(x)
print(x.shape)

y = np.array([np.arange(1,11),                       #내일의 주가, 원유가격
             np.arange(1,2,0.1),
             np.arange(9,-1,-1)]
             ).T 
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
model.add(Dense(3))

#compile & fit
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=200,batch_size=1,verbose=2)

#evaluate & predict
loss = model.evaluate(x,y)
result = model.predict([10])
print(f"LOSS: {loss}\nRESULT: {result}")

# LOSS: 3.691343563282101e-13
# RESULT: [[11.000001   2.        -0.9999994]] 