from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Conv2D, Flatten
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping
import time

#data
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,random_state=333)
print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")
# x_train.shape=(397, 10)
# x_test.shape=(45, 10)
# y_train.shape=(397,)
# y_test.shape=(45,)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler().fit(x_train)    #최솟값을 0 최댓값을 1로 스케일링
# scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
scaler = RobustScaler().fit(x_train)    #

x_train = scaler.transform(x_train).reshape(x_train.shape[0],5,2,1)
x_test = scaler.transform(x_test).reshape(x_test.shape[0],5,2,1)

#model
# model = Sequential()
# model.add(Dense(16,input_dim=10,activation='relu'))
# model.add(Dense(8))
# model.add(Dropout(0.3))
# model.add(Dense(4))
# model.add(Dense(2))
# model.add(Dropout(0.4))
# model.add(Dense(1))

model = Sequential()
model.add(Conv2D(10,(2,2), padding='same',input_shape=x_train.shape[1:]))
model.add(Conv2D(10, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Dense(1))

#compile & fit
start_time = time.time()
model.compile(loss='mse',optimizer='adam',metrics=['mse'])
es = EarlyStopping(monitor='val_loss',mode='min',patience=20,verbose=1,restore_best_weights=True)
from keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(monitor='val_loss',mode='min',save_best_only=True,
                      filepath="c:/_data/_save/MCP/diabets/keras28_"+"{epoch:04d}{val_loss:.4f}.hdf5")
hist = model.fit(x_train,y_train,epochs=1234,batch_size=16,validation_split=0.3,verbose=2)#,callbacks=[es,mcp])
end_time = time.time()
#evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=2)
y_predict = model.predict(x_test,verbose=0)
r2 = r2_score(y_test,y_predict)

print(f"Time: {round(end_time-start_time,2)}sec")
print(f"{loss=}\n{r2=}")

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
plt.title("당뇨병")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(hist.history['loss'],c='red',label='loss',marker='.')
plt.plot(hist.history['val_loss'],c='green',label='val_loss',marker='.')
plt.legend()
plt.grid()
plt.show()

# CPU Time: 33.29sec
# GPU Time: 59.25sec

# Time: 56.37sec
# loss=[4192.205078125, 4192.205078125]
# r2=0.2783869448149566