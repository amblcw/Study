from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Conv2D, Flatten
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import time

#data
path = "C:\\_data\\DACON\\따릉이\\"
train_csv = pd.read_csv(path+"train.csv",index_col=['id'])  
test_csv = pd.read_csv(path+"test.csv",index_col=0)         
submission_csv = pd.read_csv(path+"submission.csv")

train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'],axis=1) #count 를 드랍, axis=0은 행, axis=1은 열
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,shuffle=False,random_state=333)
print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")
# x_train.shape=(1313, 9)
# x_test.shape=(146, 9)
# y_train.shape=(1313,)
# y_test.shape=(146,)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler().fit(x_train)    #최솟값을 0 최댓값을 1로 스케일링
# scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
scaler = RobustScaler().fit(x_train)    #

x_train = scaler.transform(x_train).reshape(x_train.shape[0],3,3,1)
x_test = scaler.transform(x_test).reshape(x_test.shape[0],3,3,1)
test_csv = scaler.transform(test_csv).reshape(test_csv.shape[0],3,3,1)

#model
# model = Sequential()
# model.add(Dense(512,input_dim=9,activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(Dense(512,activation='relu'))
# model.add(Dense(256,activation='relu'))
# model.add(Dense(256,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(128,activation='relu'))
# model.add(Dense(64,activation='relu'))
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
es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=1024,restore_best_weights=True)
from keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(monitor='val_loss',mode='min',save_best_only=True,
                      filepath="c:/_data/_save/MCP/dacon_ddarung/k28_"+"{epoch:04d}{val_loss:.4f}.hdf5")
hist = model.fit(x_train,y_train,epochs=1234,batch_size=16,validation_split=0.35,verbose=2,callbacks=[es,mcp])
end_time = time.time()
#evaluate & predeict
loss = model.evaluate(x_test,y_test,verbose=0)
y_predict = model.predict(x_test,verbose=0)
y_submit = model.predict(test_csv,verbose=0)

import datetime
dt = datetime.datetime.now()
submission_csv['count'] = y_submit
submission_csv.to_csv(path+f"submission_{dt.day}day{dt.hour}-{dt.minute}_loss{loss[0]}.csv",index=False)

r2 = r2_score(y_test,y_predict)

print(f"Time: {round(end_time-start_time,2)}sec")
print(f"{loss=}\n{r2=}")

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False
plt.title('따릉이')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(hist.history['loss'],label='loss',color='red',marker='.')
plt.plot(hist.history['val_loss'],label='val_loss',color='blue',marker='.')
plt.legend()
plt.show()

# pationce = epo
# loss=1462.5213623046875
# r2=0.7501235084394295

# Epoch 1554: early stopping
# loss=1286.423828125
# r2=0.7802103365358142

# Epoch 2622: early stopping
# loss=1457.3612060546875
# r2=0.75100511942968

# MinMaxScaler
# loss=[1614.5137939453125, 1614.5137939453125]
# r2=0.7241550825071836

# StandardScaler
# loss=[1492.581298828125, 1492.581298828125]
# r2=0.744987662488442

# MaxAbsScaler
# loss=[1119.25048828125, 1119.25048828125]
# r2=0.8087724634833824

# RobustScaler
# loss=[1260.279296875, 1260.279296875]
# r2=0.7846772156960977

# Dropout
# loss=[1309.819091796875, 1309.819091796875]
# r2=0.7762131732898645

# CPU Time: 101.72sec
# GPU Time: 125.57sec

# CNN
# Time: 127.82sec
# loss=[1575.3001708984375, 1575.3001708984375]
# r2=0.7308548936135482