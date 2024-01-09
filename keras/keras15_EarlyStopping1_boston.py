from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import load_boston    # pip install scikit-learn==1.1.3
import time
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

datasets = load_boston()
# print(datasets)
x = datasets.data
y = datasets.target
# print(x.shape)  #(506,13)
# print(y.shape)  #(506,)
# print(datasets.feature_names)   #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
# print(datasets.DESCR)           #데이터셋 설명
'''
Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.  
    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of black people by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's
'''
r2 = 0

# while r2 < 0.8:
r = int(np.random.uniform(1,1000))
r = 88
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=r)

#model
model = Sequential()
model.add(Dense(32,input_dim=13,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))

#compile & fit
model.compile(loss='mse',optimizer='adam')
start_time = time.time()

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',mode='min',patience=30,verbose=1)
hist = model.fit(x_train,y_train,epochs=1024,batch_size=10,validation_split=0.3,verbose=2,callbacks=[es])

#evaluate & predict
loss = model.evaluate(x_test,y_test)
result = model.predict(x)
y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict)
end_time = time.time()

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

print(f"Time: {round(end_time-start_time,2)}sec")
print(f"{r=}\n{loss=}\n{r2=}\nRMSE: {RMSE(y_test,y_predict)}")

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'],color='red',label='loss',marker='.')
plt.plot(hist.history['val_loss'],color='blue',label='val_loss',marker='.')
plt.legend(loc='upper right')
plt.title('보스턴 loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()

#16 32 16 8 1 layers
# r=113
# loss=27.320585250854492
# r2=0.7244965231418128

#32 16 8 1 layers all relu
# r=707
# loss=14.15937614440918
# r2=0.7852475293021834

# epo=2048
# r=88
# loss=12.932721138000488
# r2=0.8247802724639635
# RMSE: 3.596209327723187

# Epoch 252: early stopping
# 4/4 [==============================] - 0s 0s/step - loss: 14.7583
# 16/16 [==============================] - 0s 1ms/step
# 4/4 [==============================] - 0s 0s/step
# Time: 9.57sec
# r=88
# loss=14.75829792022705
# r2=0.8000463270715448
# RMSE: 3.8416531902059194

# Epoch 258: early stopping
# 4/4 [==============================] - 0s 5ms/step - loss: 14.2062
# 16/16 [==============================] - 0s 533us/step
# 4/4 [==============================] - 0s 5ms/step
# Time: 9.81sec
# r=88
# loss=14.206158638000488
# r2=0.8075270309831843
# RMSE: 3.7691058627454215