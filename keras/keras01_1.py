import tensorflow as tf # tensorflow 를 tf로 줄여 사용
print(tf.__version__)   # 2.15.0

from keras.models import Sequential
from keras.layers import Dense
import numpy as np      

#1 데이터
x = np.array([1,2,3]) #tf 는 list를 그대로 사용하지 못함, np.array로 변환
y = np.array([1,2,3])

#2 모델구성
model = Sequential()                # 순차적 모델
model.add(Dense(1,input_dim = 1))   # 1차원 모델로 생성, y=wx+b | Dense(y차원,intput+dim=x차원)

#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #편차를 제곱하여 loss계산, 어지간한건 adam쓰면 성능이 잘 나오니 건드리지 않음
model.fit(x,y, epochs=100)                  #fit = 훈련시키기, epochs = 훈련횟수,  weight값 생성

#4 평가, 예측
loss = model.evaluate(x,y)                  #loss 값 확인
print(f"loss = {loss}")
result = model.predict([4])                 #예측 테스트
print(f"4's result = {result}")
