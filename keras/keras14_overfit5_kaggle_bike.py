from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import time
import math
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

#data
path = "C:\\_data\\KAGGLE\\bike-sharing-demand\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submission_csv = pd.read_csv(path+"sampleSubmission.csv")

x = train_csv.drop(['casual','registered','count'],axis=1)
y = train_csv['count']

print(x.shape, y.shape)

r = int(np.random.uniform(1,1000))
r=164
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,random_state=r)

# x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size=0.7,shuffle=False)

print(f"{x_train.shape=},{x_test.shape=}")


#model
model = Sequential()
model.add(Dense(32,input_dim=8,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1,activation='relu'))


#compile & fit
model.compile(loss='mse',optimizer='adam')
hist = model.fit(x_train,y_train,epochs=350,batch_size=32,verbose=2,validation_split=0.3)

#evaluate & predict 
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
y_submit = model.predict(test_csv)

print(f"{r=}\n{loss=}\n{r2=}")
time.sleep(1.5)


#### CSV파일 생성 ####
submission_csv['count'] = y_submit
dt = datetime.datetime.now()
# submission_csv.to_csv(path+f"submission_{dt.day}day{dt.hour}-{dt.minute}.csv",index=False)
submission_csv.to_csv(path+f"submission_{dt.hour}-{dt.minute}_loss{loss}.csv",index=False)


#### 음수 개수와 RMSLE출력 ####
num_of_minus = submission_csv[submission_csv['count']<0].count()
# print(num_of_minus['count'])

def RMSLE(y_test,y_predict):
    return np.sqrt(mean_squared_log_error(y_test,y_predict))

if num_of_minus['count'] == 0:    
    print("RMSLE: ",RMSLE(y_test,y_predict))
else:
    print("음수갯수: ",num_of_minus['count'])
    for i in range(len(y_submit)):
        if y_submit[i] < 0:
            y_submit[i] = 0
    
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.title('kaggle bike')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(hist.history['loss'],label='loss',color='red',marker='.')
plt.plot(hist.history['val_loss'],label='val_loss',color='blue',marker='.')
plt.grid()
plt.legend()
plt.show()