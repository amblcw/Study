from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
import numpy as np

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]
              ]).reshape(-1,3,1)

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_predict = np.array([50,60,70]).reshape(1,3,1)

print(x.shape, y.shape, x_predict.shape)

model = Sequential()
model.add(LSTM(1024, input_shape=(3,1), return_sequences=True, dropout=0.05, activation='relu'))
# model.add(LSTM(512,dropout=0.05, return_sequences=True, activation='relu'))
model.add(LSTM(512,dropout=0.05, activation='relu'))
# model.add(Dense(512))#,activation='relu'))
# model.add(Dense(512))#,activation='relu'))
# model.add(Dropout(0.01))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(loss='mse',optimizer='adam')
es = EarlyStopping(monitor='val_loss',mode='auto',patience=1024,restore_best_weights=True)
hist = model.fit(x,y,epochs=8192,batch_size=1,validation_split=0.1,verbose=2,callbacks=[es])

loss = model.evaluate(x,y)
y_predict = model.predict(x_predict)

print(f"LOSS: {loss}\ny_predict: {y_predict}")

# LOSS: 0.14223673939704895
# y_predict: [[81.18131]]

# LOSS: 5.638460636138916
# y_predict: [[89.72446]]