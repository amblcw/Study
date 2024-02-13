from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
import numpy as np
import time

N_SPLIT = 5

param = [
        {'n_jobs': [-1],'n_estimators' : [100,200], 'max_depth':[6,10,12], 'min_samples_leaf':[3,10]}, # 12
        {'n_jobs': [-1],'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10]},                       # 16
        {'n_jobs': [-1],'min_samples_leaf': [3,5,7,10], 'min_samples_split':[2,3,5,10]},               # 16
        {'n_jobs': [-1],'min_samples_split': [2,3,5,10]},                                              # 4
        {'n_jobs': [-1], 'min_samples_split':[2,3,5,10]}                                # 4 총 52
    ]

def m13_classifier(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=333)
    k_fold = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=333)
    
    model = RandomizedSearchCV(RandomForestClassifier(), param, cv=k_fold, refit=True, n_jobs=-1, n_iter=10 ,random_state=333, verbose=1)   # n_iter 기본값 10
    
    st = time.time()
    model.fit(x_train,y_train)
    et = time.time()
    
    y_predict = model.best_estimator_.predict(x_test)
    acc = accuracy_score(y_test, y_predict)
    print("acc : ", acc)
    print("time: ", et-st, "sec")
    
def m13_regressor(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=333)
    
    model = RandomizedSearchCV(RandomForestRegressor(), param, cv=N_SPLIT, refit=True, n_jobs=-1, verbose=1)
    
    st = time.time()
    model.fit(x_train,y_train)
    et = time.time()
    
    y_predict = model.best_estimator_.predict(x_test)
    r2 = r2_score(y_test, y_predict)
    print("r2 : ", r2)
    print("time: ", et-st, "sec")