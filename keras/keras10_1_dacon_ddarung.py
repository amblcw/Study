#https://dacon.io/competitions/open/235576/overview/description
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import time
import pandas as pd
import math
import datetime


import warnings
warnings.filterwarnings('ignore')

#data
path = "C:\\_data\\DACON\\ddarung\\"
train_csv = pd.read_csv(path+"train.csv",index_col=['id'])  #경로를 적을때 \n같은 경우를 방지하기 위해 \\ 나 /,//도 가능
test_csv = pd.read_csv(path+"test.csv",index_col=0)         #index_col= 을 해줘야 id를 인덱스로 사용, 0 또는 헤더명
submission_csv = pd.read_csv(path+"submission.csv")

# print(f"{train_csv.shape=}\n{test_csv.shape=}\n{submission_csv.shape=}")
# train_csv.shape=(1459, 10)
# test_csv.shape=(715, 9)
# submission_csv.shape=(715, 2)


#### 결측치 처리 1. 제거 #### 나중에 드랍 말고 다른 것도 적용해보기
# print(train_csv.isnull().sum()) 아래와 같음
print(train_csv.isna().sum())
print(test_csv.isna().sum())
# train_csv = train_csv.dropna()




nan_index = np.where(np.isnan(train_csv))

row, colum = nan_index

for i in range(len(row)):
    # print(f"({row[i]},{colum[i]})",train_csv.iloc[row[i],colum[i]])
    pre = train_csv.iloc[row[i]-1,colum[i]]
    next = train_csv.iloc[row[i]+1,colum[i]]
    train_csv.iloc[row[i],colum[i]] = (pre + next)/2
    # print(f"({row[i]},{colum[i]})",train_csv.iloc[row[i],colum[i]])

nan_index2 = np.where(np.isnan(test_csv))

row2, colum2 = nan_index2

for i in range(len(row2)):
    # print(f"({row[i]},{colum[i]})",train_csv.iloc[row[i],colum[i]])
    pre = test_csv.iloc[row2[i]-1,colum2[i]]
    next = test_csv.iloc[row2[i]+1,colum2[i]]
    test_csv.iloc[row2[i],colum2[i]] = (pre + next)/2

print(train_csv.isna().sum(), test_csv.isna().sum())


train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())



x = train_csv.drop(['count'],axis=1) #count 를 드랍, axis=0은 행, axis=1은 열
y = train_csv['count']

r2=0
# while r2<0.68:
r = int(np.random.uniform(1,1000))
r = 97
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,random_state=r)
print(f"{x_train.shape=} , {y_train.shape=}")   #x_train.shape=(1167, 9) , y_train.shape=(1167,)

#model
model = Sequential()
# model.add(Dense(1,input_dim=9))
model.add(Dense(16,input_dim=9))
model.add(Dense(32))
model.add(Dense(48))
model.add(Dense(64))
model.add(Dense(96))
model.add(Dense(128))
model.add(Dense(198))
model.add(Dense(256))
model.add(Dense(198))
model.add(Dense(128))
model.add(Dense(96))
model.add(Dense(64))
model.add(Dense(48))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#compile & fit
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=1000,batch_size=32,verbose=2)

#evaluate & predict
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
y_submit = model.predict(test_csv)

print(f"{r=}\nRMSE: {math.sqrt(loss)}\n{r2=}")
time.sleep(1.5)
    

# print(y_submit)
# print(y_submit.shape)   #(751, 1)

#### csv 만들기 ####
submission_csv['count'] = y_submit
# print(submission_csv)

dt = datetime.datetime.now()
# print(dt.day,dt.hour,dt.minute)
submission_csv.to_csv(path+f"submission_{dt.day}day{dt.hour}-{dt.minute}.csv",index=False)

# r=97
# RMSE: 45.58303131803133
# r2=0.6803156365354164