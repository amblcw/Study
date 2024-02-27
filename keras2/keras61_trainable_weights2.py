from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
print(tf.__version__)   # 2.9.0
tf.random.set_seed(777) # 이쪽이 가중치 초기화에 영향
np.random.seed(777)

x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

#########################################
model.trainable = False # ★★★
# model.trainable = True # ★★★
#########################################
# model.summary()
print(model.weights)
print('=====================')

# compile & fit
model.compile(loss='mse',optimizer='adam')
model.fit(x,y, batch_size=1, epochs=1000,verbose=0)

# eval
print("loss: ",model.evaluate(x,y))
y_predict = model.predict(x)
print(y_predict)