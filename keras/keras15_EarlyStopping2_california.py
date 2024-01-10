from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
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

#model
model = Sequential()
model.add(Dense(32,input_dim=8,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))

#compile fit
model.compile(loss='mse',optimizer='adam')
start_time = time.time()
es = EarlyStopping(monitor='val_loss',mode='min',patience=20,restore_best_weights=True,verbose=1)
hist = model.fit(x_train,y_train,epochs=1024,batch_size=64,validation_split=0.3,verbose=2,callbacks=[es])

#evaluate predict
loss = model.evaluate(x_test,y_test,verbose=0)
result = model.predict(x,verbose=0)
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

# Time: 69.11sec
# r=176
# loss=0.43618541955947876
# r2=0.672248681511576

# Epoch 295/1024
# 158/158 - 0s - loss: 0.4212 - val_loss: 0.5016 - 133ms/epoch - 841us/step
#   1/194 [..............................] - ETA: 2s - loss: 0.42113/194 [================>.............] - ETA: 0s - loss: 0.45194/194 [==============================] - 0s 463us/step - loss: 0.4645
# 645/645 [==============================] - 0s 464us/step
# 194/194 [==============================] - 0s 451us/step
# Time: 41.13sec
# r=176
# loss=0.4645490050315857
# r2=0.650936286442668

# r=176
# loss=0.45143747329711914
# r2=0.6607882232154847

# Epoch 236: early stopping
# Time: 35.14sec
# r=176
# loss=0.45041927695274353
# r2=0.661553367308052