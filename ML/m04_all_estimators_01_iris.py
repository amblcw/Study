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

import warnings
warnings.filterwarnings(action='ignore')

#data
x, y = load_iris(return_X_y=True)
# print(f"{x.shape=}, {y.shape=}")        #x.shape=(150, 4), y.shape=(150,)
# print(np.unique(y, return_counts=True)) #array([0, 1, 2]), array([50, 50, 50])

x_train , x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=333,stratify=y)

#model
from sklearn.utils import all_estimators

all_algorithms = all_estimators(type_filter='classifier')
# all_algorithms = all_estimators(type_filter='regressor')
print(len(all_algorithms))  # 41(분류) 55(회귀) 

error_list = []
for name, algorithm in all_algorithms:
    try:
        model = algorithm()
    except Exception as e:
        print(f"{name:30} ERROR")
        error_list.append(e)
        continue
        
    model.fit(x_train,y_train)
    acc = model.score(x_test,y_test)
    print(f"{name:30} ACC: {acc:.4f}")
print('error_list: \n',error_list)

# ACC list:  [0.93, 0.5, 0.93, 0.93, 0.9, 0.9]
# Best ML:  LinearSVC