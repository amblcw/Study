from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import sklearn.preprocessing



#data
x, y = load_iris(return_X_y=True)
# print(f"{x.shape=}, {y.shape=}")        #x.shape=(150, 4), y.shape=(150,)
# print(np.unique(y, return_counts=True)) #array([0, 1, 2]), array([50, 50, 50])

x_train , x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=333,stratify=y)

#model
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
    acc = round(model.score(x_test,y_test),2)
    # y_predict = model.predict(x_test)
    # acc = accuracy_score(y_test,y_predict)
    acc_list.append(acc)
    
#결과값 출력
print("ACC list: ", acc_list)
print("Best ML: ",model_names[acc_list.index(max(acc_list))])

# ACC list:  [0.93, 0.5, 0.93, 0.93, 0.9, 0.9]
# Best ML:  LinearSVC