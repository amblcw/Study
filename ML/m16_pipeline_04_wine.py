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
import warnings
warnings.filterwarnings(action='ignore')

datasets = load_wine()
x = datasets.data
y = datasets.target

# print(x.shape,y.shape)  #(178, 13) (178,)
# print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48
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
    
param = {'max_depth': 6, 'min_samples_leaf': 3}
m15_classifier(x, y, param)
# r=398
# LOSS: 0.13753800094127655
# ACC:  1.0(1.0 by loss[1])

# Best result : SVC`s 1.0000

# ACC:  [0.72222222 0.72222222 0.61111111 0.62857143 0.74285714]
# 평균 ACC: 0.6854

# 최적의 매개변수:  RandomForestClassifier(max_depth=6, min_samples_leaf=3)
# 최적의 파라미터:  {'max_depth': 6, 'min_samples_leaf': 3}
# best score:  0.9875
# model score:  1.0
# ACC:  1.0
# y_pred_best`s ACC: 1.0
# time: 2.86sec

# Random
# acc :  0.7251908396946565
# time:  2.465505361557007 sec

# n_iterations: 2
# n_required_iterations: 4
# n_possible_iterations: 2
# min_resources_: 30
# max_resources_: 142
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 52
# n_resources: 30
# Fitting 5 folds for each of 52 candidates, totalling 260 fits
# ----------
# iter: 1
# n_candidates: 18
# n_resources: 90
# Fitting 5 folds for each of 18 candidates, totalling 90 fits
# acc :  0.9722222222222222
# time:  26.515902996063232 sec

# acc :  0.9722222222222222
# time:  0.06301736831665039 sec