from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#data
x = np.array([range(10)])
print(x)                    #[[0 1 2 3 4 5 6 7 8 9]] (1,10)
print(x.shape)

x = np.array([range(1,10)])
print(x)                    #[[1 2 3 4 5 6 7 8 9]] (1,9)
print(x.shape)

x = np.array([range(10), range(21,31), range(201,211)]).T   #주가, 금리, 환율
print(x)
print(x.shape)

y = np.array([np.arange(1,11),                       #내일의 주가, 원유가격
             np.arange(1,2,0.1)]
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
model.add(Dense(6))
model.add(Dense(2))

#compile & fit
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=500,batch_size=1,verbose=2)

#evaluate & predict
loss = model.evaluate(x,y)
result = model.predict([[10,31,211]])
print(f"LOSS: {loss} \nRESULT: {result}")

# LOSS: 1.460961074339906e-10
# RESULT: [[11.000001   2.0000029]]