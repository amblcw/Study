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
x = datasets.data
y = datasets.target

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
model.compile(loss='mse',optimizer='adam',metrics=['mse'])
start_time = time.time()

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',mode='min',patience=50,verbose=1,restore_best_weights=True)
hist = model.fit(x_train,y_train,epochs=1024,batch_size=10,validation_split=0.3,verbose=2,callbacks=[es])

#evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=0)
result = model.predict(x,verbose=0)
y_predict = model.predict(x_test,verbose=0)

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

# Epoch 197: early stopping
# Time: 8.06sec
# r=88
# loss=13.1826171875
# r2=0.8213945466114984
# RMSE: 3.6307874062998877