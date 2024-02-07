from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import load_boston    # pip install scikit-learn==1.1.3
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.svm import LinearSVR

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


# scaler = MinMaxScaler().fit(x_train)    #최솟값을 0 최댓값을 1로 스케일링
scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
# scaler = RobustScaler().fit(x_train)    #

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(np.min(x_train),np.max(x_train))
# print(np.min(x_test),np.max(x_test))

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

# ACC list:  [0.7791, 0.7918, 0.873, 0.8077, 0.902]
# Best ML:  RandomForestRegressor