from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#data
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
             [9,8,7,6,5,4,3,2,1,0]]
             ).T
# x.T == x.transpose(x) == x.swapaxes(x,0,1)
y = np.array([1,2,3,4,5,6,7,8,9,10])
print(x.shape)  # (10,2)
print(y.shape)  # (10,) 데이터 개수 맞춰주기

#model generation
model = Sequential()
model.add(Dense(10,input_dim=3))
model.add(Dense(30))
model.add(Dense(60))
model.add(Dense(100))
model.add(Dense(300))
model.add(Dense(100))
model.add(Dense(60))
model.add(Dense(30))  
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))

#complie & fit
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=300,batch_size=1,verbose=2)

#evaluate & predict
loss = model.evaluate(x,y)
result = model.predict([[10,1.3,0]])
print(f"LOSS: {loss} \n RESULT: {result}")

# LOSS: 2.113509454254592e-12        
#  RESULT: [[9.999998]]