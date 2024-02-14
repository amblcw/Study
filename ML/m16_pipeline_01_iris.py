from sklearn.datasets import load_iris

x, y = load_iris(return_X_y=True)

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
    
param = {'max_depth': 6, 'min_samples_leaf': 3, 'n_estimators': 200}
m15_classifier(x, y, param)
    
# 최적의 매개변수:  RandomForestClassifier(max_depth=6, min_samples_leaf=3, n_estimators=200)
# 최적의 파라미터:  {'max_depth': 6, 'min_samples_leaf': 3, 'n_estimators': 200}
# best score:  0.9555555555555555
# model score:  1.0
# ACC:  1.0
# y_pred_best`s ACC: 1.0
# time: 2.85sec

# acc :  0.9666666666666667
# time:  1.7793960571289062 sec

# n_iterations: 3
# n_required_iterations: 4
# n_possible_iterations: 3
# min_resources_: 10
# max_resources_: 120
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 52
# n_resources: 10
# Fitting 5 folds for each of 52 candidates, totalling 260 fits
# ----------
# iter: 1
# n_candidates: 18
# n_resources: 30
# Fitting 5 folds for each of 18 candidates, totalling 90 fits
# ----------
# iter: 2
# n_candidates: 6
# n_resources: 90
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# R2 :  0.9491690204159742
# time:  22.16383123397827 sec

# acc :  0.9666666666666667
# time:  0.10776543617248535 sec