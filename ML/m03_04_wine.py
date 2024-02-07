from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

datasets = load_wine()
x = datasets.data
y = datasets.target

# print(x.shape,y.shape)  #(178, 13) (178,)
# print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48

r = int(np.random.uniform(1,1000))
r = 398
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=r,stratify=y)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler().fit(x_train)    #최솟값을 0 최댓값을 1로 스케일링
# scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
scaler = RobustScaler().fit(x_train)    #

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model_list = [LinearSVC(), 
              Perceptron(), 
              LogisticRegression(), 
              KNeighborsClassifier(), 
              DecisionTreeClassifier(), 
              RandomForestClassifier(),
              ]
model_names = ['LinearSVC','Perceptron','LogisticRegression','KNeighborsClassifier','DecisionTreeClassifier','RandomForestClassifier']
acc_list = []

for model in model_list:
    #compile & fit
    model.fit(x_train,y_train)

    #evaluate & predict
    acc = round(model.score(x_test,y_test),4)
    # y_predict = model.predict(x_test)
    # acc = accuracy_score(y_test,y_predict)
    acc_list.append(acc)
    
#결과값 출력
print("ACC list: ", acc_list)
print("Best ML: ",model_names[acc_list.index(max(acc_list))])

# ACC list:  [1.0, 1.0, 1.0, 0.9167, 0.9167, 1.0]
# Best ML:  LinearSVC