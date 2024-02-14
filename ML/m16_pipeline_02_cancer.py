import numpy as np
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import LinearSVC

import warnings
warnings.filterwarnings(action='ignore')

# data
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)
x = datasets.data
y = datasets.target 
df_y = pd.DataFrame(y)

# print(df_y)
# print(x,y,x.shape,y.shape,sep='\n')
print(np.unique(y,return_counts=True)) #(array([0, 1]), array([212, 357], dtype=int64))
zero_num = len(y[np.where(y == 0)]) #y[np.where(조건)]은 조건에 맞는 값들의 인덱스 리스트를 반환
one_num = len(y[np.where(y == 1)])
print(f"0: {zero_num}, 1: {one_num}")
print(df_y.value_counts()) #pandas 요소 개수 세기
print(pd.value_counts(y))  #위와 동일

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import time

def m15_classifier(x,y,param, **kwargs):
    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=333)

    model = make_pipeline(MinMaxScaler(),RandomForestClassifier(**param))
    
    st = time.time()
    model.fit(x_train,y_train)
    et = time.time()
    
    acc = model.score(x_test, y_test)
    print("acc : ", acc)
    print("time: ", et-st, "sec")
    
param = {'max_depth': 12, 'min_samples_leaf': 3}
m15_classifier(x, y, param)


# 최적의 매개변수:  RandomForestClassifier(max_depth=12, min_samples_leaf=3)
# 최적의 파라미터:  {'max_depth': 12, 'min_samples_leaf': 3}
# best score:  0.9609366076527699
# model score:  0.9824561403508771
# ACC:  0.9824561403508771
# y_pred_best`s ACC: 0.9824561403508771
# time: 3.31sec

# Random
# acc :  0.9385964912280702
# time:  2.0538978576660156 sec

# n_iterations: 3
# n_required_iterations: 4
# n_possible_iterations: 3
# min_resources_: 20
# max_resources_: 455
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 52
# n_resources: 20
# Fitting 5 folds for each of 52 candidates, totalling 260 fits
# ----------
# iter: 1
# n_candidates: 18
# n_resources: 60
# Fitting 5 folds for each of 18 candidates, totalling 90 fits
# ----------
# iter: 2
# n_candidates: 6
# n_resources: 180
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# acc :  0.9473684210526315
# time:  30.55355167388916 sec

# make_pipeline
# acc :  0.9298245614035088
# time:  0.1615138053894043 sec