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

#compile & fit
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min',patience=50,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='min',verbose=1,save_best_only=False,
                      filepath="../_data/_save/MCP/keras25_MCP3.hdf5")
hist = model.fit(x_train,y_train,epochs=1024,batch_size=10,validation_split=0.3,verbose=2,callbacks=[es,mcp])
model.save("../_data/_save/keras25_3_save_model.h5")  #가중치와 모델 모두 담겨있다

#evaluate & predict
print("============ 1. 기본출력 ============")
loss = model.evaluate(x_test,y_test,verbose=0)
y_predict = model.predict(x_test,verbose=0)

r2 = r2_score(y_test,y_predict)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

print(f"{r=}\n{loss=}\n{r2=}\nRMSE: {RMSE(y_test,y_predict)}")

print("========== 2. LOAD_MODEL ==========")
model2 = load_model("../_data/_save/keras25_3_save_model.h5")
loss2 = model2.evaluate(x_test,y_test,verbose=0)
y_predict2 = model2.predict(x_test,verbose=0)

r2 = r2_score(y_test,y_predict)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

print(f"{r=}\n{loss=}\n{r2=}\nRMSE: {RMSE(y_test,y_predict2)}")
# print(hist.history['val_loss'])

# plt.rcParams['font.family'] ='Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] =False
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'],color='red',label='loss',marker='.')
# plt.plot(hist.history['val_loss'],color='blue',label='val_loss',marker='.')
# plt.legend(loc='upper right')
# plt.title('보스턴 loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.grid()
# plt.show()

# StandardScaler
# loss=[4.916057586669922, 1.7392468452453613]
# r2=0.9333945146886241
# RMSE: 2.217218335403849

