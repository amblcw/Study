from tensorflow.python.keras.models import Sequential
# from tensorflow.keras.models import Sequential
# from keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D

model = Sequential()
# model.add(Dense(10, input_shape=(3,)))
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(10,10,1))) #Conv2D(다음레이어로 전달할 출력수, kernel_size=몇개씩 자를것인가, input_shape(Dense때와 동일)), kernel_size는 가중치의 shape
model.add(Dense(5))
model.add(Dense(1))