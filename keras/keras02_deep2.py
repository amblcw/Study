from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#데이터 전처리
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#모델구성
model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(6))
model.add(Dense(100))
model.add(Dense(1000))
model.add(Dense(5000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(4)) 
model.add(Dense(4)) 
model.add(Dense(4)) 
model.add(Dense(4)) 
model.add(Dense(4)) 
model.add(Dense(4)) 
model.add(Dense(2))
model.add(Dense(1))

#컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100)

#평가, 예측
loss = model.evaluate(x,y)
print(f"LOSS : {loss}")
result = model.predict([7])
print(f"RESULT : {result}")


# 1/1 [==============================] - 0s 194ms/step - loss: 0.3296
# LOSS : 0.3296014368534088
# 1/1 [==============================] - 0s 130ms/step
# RESULT : [[6.96025]]