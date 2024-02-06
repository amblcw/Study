from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np
import keras
import random as rn

rn.seed(333)
tf.random.set_seed(123) # 텐서 2.9에서는 적용되나 2.15에서는 적용 안됨
np.random.seed(123)

print(tf.__version__)   # 2.9.0
print(np.__version__)   # 1.26.3
print(keras.__version__)# 2.9.0

# data
x = np.array([1,2,3])
y = np.array([1,2,3])

# model
model = Sequential()
model.add(Dense(5, 
                # kernel_initializer='zeros', 
                # bias_initializer='zeros',
                input_dim=1))
model.add(Dense(5))
model.add(Dense(1))

# compile fit
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100)

# evaluate
loss = model.evaluate(x,y, verbose=0)
result = model.predict([4])
print(f"loss: {loss}\n4 predict: {result}")

# loss: 21.64821434020996
# 4 predict: [[-4.6855907]]