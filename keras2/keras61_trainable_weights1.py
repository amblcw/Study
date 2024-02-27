from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
print(tf.__version__)   # 2.9.0
# tf.random.set_seed(777) # 이쪽이 가중치 초기화에 영향
# np.random.seed(777)

x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

print(model.weights)
'''
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.47288632, -0.78825045,  1.2209238 ]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[-0.1723156 ,  0.5125139 ],
       [ 0.41434443, -0.8537577 ],
       [ 0.5188304 , -0.91461056]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
array([[ 1.1585606],
       [-0.4251585]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]

1번줄의 kernel은 가중치를 의미한다
'''
print('=====================')
print(model.trainable_weights)
print('=====================')
print(len(model.weights))   # 한 레이어에 가중치리스트 1개, 바이어스 리스트 1개 총 2개씩 있음
print(len(model.trainable_weights))

#########################################
model.trainable = False # ★★★
#########################################

print(len(model.weights))   # 한 레이어에 가중치리스트 1개, 바이어스 리스트 1개 총 2개씩 있음
print(len(model.trainable_weights))
print('=====================')
print(model.trainable_weights)
print('=====================')
model.summary()
# Total params: 17
# Trainable params: 0
# Non-trainable params: 17