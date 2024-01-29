import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout, LSTM
from keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE

# data
datasets = np.array(range(1,11))

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]]).reshape(-1,3,1)
y = np.array([4,5,6,7,8,9,10])

print(x.shape,y.shape) # (7, 3, 1) (7,)

# model
model = Sequential()
model.add(LSTM(units=10,input_shape=(3,1),activation='relu')) # input: (batch_size, timesteps, features).
# model.add(SimpleRNN(units=1024,input_shape=(3,1),activation='relu')) # input: (batch_size, timesteps, features).
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

model.summary()

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  lstm (LSTM)                 (None, 10)                480

#  dense (Dense)               (None, 7)                 77

#  dense_1 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 565
# Trainable params: 565
# Non-trainable params: 0