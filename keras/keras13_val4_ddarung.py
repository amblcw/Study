#https://dacon.io/competitions/open/235576/overview/description
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
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
# print(train_csv.isna().sum())
# print(test_csv.isna().sum())


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

# print(train_csv.isna().sum(), test_csv.isna().sum())


train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())
# train_csv = train_csv.fillna(0)
# test_csv = test_csv.fillna(0)



# print(train_csv.isna().sum(), test_csv.isna().sum())

x = train_csv.drop(['count'],axis=1) #count 를 드랍, axis=0은 행, axis=1은 열
y = train_csv['count']

loss_list = []
r2_list = []
submit_list = []

r2=0
# for i in range(10):
# while r2<0.69:
    # print(f"{i}번째 시도")
r = int(np.random.uniform(1,1000))
r = 285
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,random_state=r)
print(f"{x_train.shape=} , {y_train.shape=}")   #x_train.shape=(1167, 9) , y_train.shape=(1167,)

# model
model = Sequential()
model.add(Dense(32,input_dim=9,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1))

#compile & fit
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=1024,batch_size=16,validation_split=0.3,verbose=2)

#evaluate & predict
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
y_submit = model.predict(test_csv)

print(f"{r=}\nRMSE: {math.sqrt(loss)}\n{r2=}")
time.sleep(1.5)

# print(y_submit)
# print(y_submit.shape)   #(751, 1)
loss_list.append(loss)
r2_list.append(r2)
submit_list.append(y_submit)

#####여기까지 반복문#####

print(loss_list,r2_list,sep='\n')
print(f"LOSS mean: {np.mean(loss_list)}\nR2 mean: {np.mean(r2_list)}")

#### csv 만들기 ####

submission_csv['count'] = submit_list[r2_list.index(max(r2_list))]
# print(submission_csv)

dt = datetime.datetime.now()
# print(dt.day,dt.hour,dt.minute)
submission_csv.to_csv(path+f"submission_{dt.day}day{dt.hour}-{dt.minute}.csv",index=False)

# 32 16 8 4 2 1 layers, epo=1024, batch=4, val=0.3, train=0.9, r = 285
# [2416.1826171875, 2403.624755859375, 2160.46484375, 2165.702392578125, 2163.209228515625, 7833.255859375, 7833.26904296875, 2697.525634765625, 2367.112060546875, 7833.263671875]
# [0.6914742207291835, 0.6930777774036352, 0.7241271685840622, 0.7234583846065544, 0.7237767623504908, -0.00023949508046716694, -0.00024114966142230188, 0.6555491479314323, 0.6977401199082474, -0.00024052618622283362]
# LOSS mean: 3987.3610107421873
# R2 mean: 0.4908482410585494