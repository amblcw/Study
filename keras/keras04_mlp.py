from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#data
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]]
             ).T
# x.T == x.transpose(x) == x.swapaxes(x,0,1)
y = np.array([1,2,3,4,5,6,7,8,9,10])
print(x.shape)  # (2,10)
print(y.shape)  # (10,)

#model
model = Sequential()
model.add(Dense(3,input_dim=2)) # (행 무시, 열 우선) input_dim에 열의 갯수만 맞추고 행은 무시
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(60))
model.add(Dense(100))
model.add(Dense(60))
model.add(Dense(30))  
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))

#compile & fit
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=300,batch_size=1,verbose=2)

#evaluate & predict
loss = model.evaluate(x,y)
result = model.predict([[10,1.3]]) # 이것 또한 입력값이기에 열을 꼭 맞춰주기 ※행무시 열우선
print(f"LOSS: {loss}")
print(f"predict about [10, 1.3]: {result}")

# LOSS: 8.76464673638111e-06
# predict about [10, 1.3]: [[10.002944]]  