from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#데이터 전처리
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#모델구성
model = Sequential()
model.add(Dense(1,input_dim = 1))

#컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=15000)

#평가, 예측
loss = model.evaluate(x,y)
print(f"LOSS : {loss}")
result = model.predict([7])
print(f"RESULT : {result}")

# 2차 데이터
new_x = []
new_y = []
for i in range(10,60):
    n = model.predict([i/10]).tolist()
    new_y.append(n[0])
    new_x.append(i/10)

print(new_y)
new_x = np.array(new_x)
new_y = np.array(new_y)

#모델구성
model = Sequential()
model.add(Dense(1,input_dim = 1))

#컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(new_x,new_y,epochs=15000)

#평가, 예측
loss = model.evaluate(x,y)
print(f"LOSS : {loss}")
result = model.predict([1,2,3,4,5,6,7])
print(f"RESULT : {result}")
# print(new_y)

# epochs = 15000
# LOSS : 0.3238094747066498
# 1/1 [==============================] - 0s 33ms/step
# RESULT : [[1.1428571]
#  [2.0857143]
#  [3.0285714]
#  [3.9714286]
#  [4.9142857]
#  [5.857143 ]
#  [6.8      ]]