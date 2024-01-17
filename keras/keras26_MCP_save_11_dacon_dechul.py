#25-5 copy
from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import load_boston    # pip install scikit-learn==1.1.3
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import datetime
dt = datetime.datetime.now()



datasets = load_boston()
x = datasets.data
y = datasets.target

r = int(np.random.uniform(1,1000))
r = 88
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=r)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler().fit(x_train)    #최솟값을 0 최댓값을 1로 스케일링
scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
# scaler = RobustScaler().fit(x_train)    #

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#model
model = Sequential()
model.add(Dense(32,input_dim=13,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))

# model.summary()

# print(type(dt),dt)   #2024-01-17 10:52:40.184440
# date = dt.strftime("%m%d_%H%M")
# print(date)
# path = "../_data/_save/MCP/"
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
# filepath = "".join([path,'k25_',date,'_',filename])
# #compile & fit
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min',patience=10,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='min',verbose=1,save_best_only=True,
                    #   filepath=filepath)
                      filepath=f"../_data/_save/MCP/k25/{dt.day}{dt.hour}_"+"{epoch:04d}-{loss:.4f}.hdf5")
hist = model.fit(x_train,y_train,epochs=1000,batch_size=10,validation_split=0.3,verbose=2,callbacks=[es,mcp])
model.save("../_data/_save/keras25_3_save_model.h5")  #가중치와 모델 모두 담겨있다


# filename2 = path+f"{dt.month:02}{dt.day:02}_{dt.hour:02}{dt.minute:02}"

#evaluate & predict
print("============ 1. 기본출력 ============")
loss = model.evaluate(x_test,y_test,verbose=0)
y_predict = model.predict(x_test,verbose=0)

r2 = r2_score(y_test,y_predict)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

print(f"{r=}\n{loss=}\n{r2=}\nRMSE: {RMSE(y_test,y_predict)}")

print(hist.history['val_loss'])

# save_best_only        =True
# restore_best_weight   =True
# 갱신된 epo만 저장, 최고가중치 마지막에 저장

# save_best_only        =True
# restore_best_weight   =False
# 갱신된 epo만 저장

# save_best_only        =False
# restore_best_weight   =True
# 모든 epo별로 전부 저장

# save_best_only        =False
# restore_best_weight   =False
# 모든 epo별로 전부 저장

# 기본출력(restore_best_weight 안되어있음)
# r=88
# loss=[6.097459316253662, 1.8482483625411987]        
# r2=0.9173882207844651
# RMSE: 2.4693033788182004

# mcp출력(마지막으로 갱신된 val_loss로)
# r=88
# loss=[6.302283763885498, 1.876548171043396]
# r2=0.9146131446215406
# RMSE: 2.510434951014859