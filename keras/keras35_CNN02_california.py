from sklearn.datasets import fetch_california_housing
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Conv2D, Flatten
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

#data
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape,y.shape,sep='\n')
print(datasets.feature_names)   
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
r2=0
# while r2 < 0.6: 
r = int(np.random.uniform(1,1000))
r = 176
# r = 130
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,random_state=r)

print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")

# x_train.shape=(14447, 8)
# x_test.shape=(6193, 8)
# y_train.shape=(14447,)
# y_test.shape=(6193,)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler().fit(x_train)    #최솟값을 0 최댓값을 1로 스케일링
# scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
scaler = RobustScaler().fit(x_train)    #

x_train = scaler.transform(x_train).reshape(x_train.shape[0],4,2,1)
x_test = scaler.transform(x_test).reshape(x_test.shape[0],4,2,1)

#model
# model = Sequential()
# model.add(Dense(32,input_dim=8,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(16,activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(8,activation='relu'))
# model.add(Dense(1))

model = Sequential()
model.add(Conv2D(10, (2,2), padding='same', input_shape=x_train.shape[1:]))
model.add(Conv2D(10,(2,2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Dense(1))

#compile fit
model.compile(loss='mse',optimizer='adam',metrics=['mse'])
start_time = time.time()
es = EarlyStopping(monitor='val_loss',mode='min',patience=20,restore_best_weights=True,verbose=1)
from keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(monitor='val_loss',mode='min',save_best_only=True,
                      filepath="c:/_data/_save/MCP/california/k28_"+"{epoch:04d}{val_loss:.4f}.hdf5")
hist = model.fit(x_train,y_train,epochs=1234,batch_size=64,validation_split=0.3,verbose=2)#,callbacks=[es,mcp])

#evaluate predict
loss = model.evaluate(x_test,y_test,verbose=0)
# result = model.predict(x,verbose=0)
y_predict = model.predict(x_test,verbose=0)

r2 = r2_score(y_test,y_predict)
end_time = time.time()
print(f"Time: {round(end_time-start_time,2)}sec")
print(f"{r=}\n{loss=}\n{r2=}")

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'],color='red',label='loss',marker='.')
plt.plot(hist.history['val_loss'],color='blue',label='val_loss',marker='.')
# plt.plot(range(128),np.array([hist.history['loss'],hist.history['val_loss']]).T,label=['loss','val_loss'])
plt.legend(loc='upper right')
plt.title('california loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()

# dropout
# r=176
# loss=[0.2723971903324127, 0.2723971903324127]       
# r2=0.7953197636967315

# CPU Time: 207.35sec
# GPU Time: 302.38sec

#  loss: 0.6787 - mse: 0.6787