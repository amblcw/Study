from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVR

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

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler().fit(x_train)    #최솟값을 0 최댓값을 1로 스케일링
# scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
scaler = RobustScaler().fit(x_train)    #

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

from sklearn.svm import SVR
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

model_list = [SVR(), 
              LinearRegression(), 
              KNeighborsRegressor(), 
              DecisionTreeRegressor(), 
              RandomForestRegressor(),
              ]
model_names = ['SVR','LinearRegression','KNeighborsRegressor','DecisionTreeRegressor','RandomForestRegressor']
loss_list = []

for model in model_list:
    #compile & fit
    model.fit(x_train,y_train)

    #evaluate & predict
    loss = round(model.score(x_test,y_test),4)
    # y_predict = model.predict(x_test)
    # acc = accuracy_score(y_test,y_predict)
    loss_list.append(loss)
    
#결과값 출력
print("ACC list: ", loss_list)
print("Best ML: ",model_names[loss_list.index(max(loss_list))])


# Epoch 455: early stopping <= best
# loss=1431.286376953125
# r2=0.7554601030711634
# model.add(Dense(512,input_dim=9,activation='relu'))
# model.add(Dense(512,activation='relu'))
# model.add(Dense(256,activation='relu'))
# model.add(Dense(256,activation='relu'))
# model.add(Dense(128,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(1))

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

# LinearSVR
# loss=0.652966147154205
# r2=0.652966147154205

# ACC list:  [0.493, 0.6613, 0.7235, 0.417, 0.784]
# Best ML:  RandomForestRegressor