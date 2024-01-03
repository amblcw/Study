from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#data
x = np.array([range(10), range(21,31), range(201,211)]).T   #주가, 금리, 환율
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
model.add(Dense(10,input_dim=3))
model.add(Dense(30))
model.add(Dense(100))
model.add(Dense(300))
model.add(Dense(100))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(3))

#compile & fit
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=200,batch_size=1,verbose=2)

#evaluate & predict
loss = model.evaluate(x,y)
result = model.predict([[10,31,211]])
print(f"LOSS: {loss}\nRESULT: {result}")

# LOSS: 8.846630025760582e-12
# RESULT: [[10.999999   2.0000055 -0.9999975]]