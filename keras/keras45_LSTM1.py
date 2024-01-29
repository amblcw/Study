import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout, LSTM
from keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
from function_package import split_x

# data
datasets = np.array(range(1,11))

# x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]]).reshape(-1,3,1)
x = split_x(datasets,3)[:-1]
y = np.array([4,5,6,7,8,9,10])

print(x)
print(x.shape,y.shape) # (7, 3, 1) (7,)

# model
model = Sequential()
model.add(LSTM(units=1024,input_shape=(3,1),activation='relu')) # input: (batch_size, timesteps, features).
# model.add(SimpleRNN(units=1024,input_shape=(3,1),activation='relu')) # input: (batch_size, timesteps, features).
model.add(Dropout(0.05))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# compile & fit
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss',mode='auto',patience=300,restore_best_weights=True)
model.fit(x,y,epochs=1000,batch_size=1,validation_split=0.1,verbose=2,callbacks=[es])

# evaluate & predict
loss = model.evaluate(x,y)
y_predict = model.predict([[[8],[9],[10]]])

print("[8,9,10] => ",y_predict)
# [8,9,10] =>  [[11.000075]]