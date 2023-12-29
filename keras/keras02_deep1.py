from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# 모델생성
model = Sequential()
model.add(Dense(10, input_dim = 1))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(1000))
model.add(Dense(2000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(6))
model.add(Dense(6))
model.add(Dense(6))
model.add(Dense(6))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))

# 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100)

# 평가, 예측
loss = model.evaluate(x,y)
print(f"LOSS : {loss}")
result = model.predict([4])
print(f"RESULT : {result}")

# 1/1 [==============================] - 0s 180ms/step - loss: 1.3342e-04
# LOSS : 0.00013341846351977438
# 1/1 [==============================] - 0s 146ms/step
# RESULT : [[4.0021353]]