from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
import keras

print(f"tf 버전: {tf.__version__}")
print(f"keras 버전: {keras.__version__}")

#데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])
#모델 생성
model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(100))
model.add(Dense(300))
# model.add(Dense(1000))
# model.add(Dense(750))
# model.add(Dense(500))
model.add(Dense(250))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(6)) 
model.add(Dense(4)) 
model.add(Dense(2))
model.add(Dense(1))

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=2, verbose=2)

#평가, 예측
loss = model.evaluate(x,y)
result = model.predict([7])
print(f"LOSS: {loss}")
print(f"7의 예측값: {result}")

# LOSS: 0.3238898515701294
# 7의 예측값: [[6.7873745]]
