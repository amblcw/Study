import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout
from keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE

# data
datasets = np.array(range(1,11))

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]]).reshape(-1,3,1)
y = np.array([4,5,6,7,8,9,10])

print(x.shape,y.shape) # (7, 3, 1) (7,)

# model
model = Sequential()
model.add(SimpleRNN(units=10,input_shape=(3,1), activation='relu')) # input: (batch_size, timesteps, features).
# model.add(SimpleRNN(16,activation='relu'))
model.add(Dropout(0.05))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

model.summary()

'''
param = units*(units + features + bias)
units가 더해지는 이유는 RNN은 이전 아웃풋을 히든레이어로 갖기 때문이다 
h = W*h(t-1) + W*x + b
'''

# compile & fit
# model.compile(loss='mse', optimizer='adam')
# es = EarlyStopping(monitor='val_loss',mode='auto',patience=300,restore_best_weights=True)
# model.fit(x,y,epochs=1000,batch_size=1,validation_split=0.1,verbose=2,callbacks=[es])

# # evaluate & predict
# loss = model.evaluate(x,y)
# y_predict = model.predict([[[8],[9],[10]]])

# print("[8,9,10] => ",y_predict)
# [8,9,10] =>  [[11.000075]]