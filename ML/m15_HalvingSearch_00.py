from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import time

'''
최소 iter 2 이상
'''


def m15_classifier(x,y,**kwargs):
    '''
    mix_resources
    '''
    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=333)
    scaler = MinMaxScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    N_SPLIT = 5

    param = [
            {'n_jobs': [-1],'n_estimators' : [100,200], 'max_depth':[6,10,12], 'min_samples_leaf':[3,10]}, # 12
            {'n_jobs': [-1],'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10]},                       # 16
            {'n_jobs': [-1],'min_samples_leaf': [3,5,7,10], 'min_samples_split':[2,3,5,10]},               # 16
            {'n_jobs': [-1],'min_samples_split': [2,3,5,10]},                                              # 4
            {'n_jobs': [-1], 'min_samples_split':[2,3,5,10]}                                # 4 총 52
        ]
    
    model = HalvingGridSearchCV(RandomForestClassifier(), param, cv=N_SPLIT, refit=True, verbose=1, factor=3)
    if 'min_resources' in kwargs:
        model = HalvingGridSearchCV(RandomForestClassifier(), param, cv=N_SPLIT, refit=True, verbose=1, factor=3, min_resources=kwargs['min_resources'])
            
    st = time.time()
    model.fit(x_train,y_train)
    et = time.time()
    
    y_predict = model.best_estimator_.predict(x_test)
    acc = accuracy_score(y_test, y_predict)
    print("acc : ", acc)
    print("time: ", et-st, "sec")
    
    
def m15_regressor(x,y, **kwargs):
    '''
    mix_resources
    '''
    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=333)
    scaler = MinMaxScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    N_SPLIT = 5

    param = [
            {'n_jobs': [-1],'n_estimators' : [100,200], 'max_depth':[6,10,12], 'min_samples_leaf':[3,10]}, # 12
            {'n_jobs': [-1],'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10]},                       # 16
            {'n_jobs': [-1],'min_samples_leaf': [3,5,7,10], 'min_samples_split':[2,3,5,10]},               # 16
            {'n_jobs': [-1],'min_samples_split': [2,3,5,10]},                                              # 4
            {'n_jobs': [-1], 'min_samples_split':[2,3,5,10]}                                # 4 총 52
        ]
    
    model = HalvingGridSearchCV(RandomForestRegressor(), param, cv=N_SPLIT, refit=True, verbose=1, factor=3)
    if 'mix_resources' in kwargs:
        model = HalvingGridSearchCV(RandomForestRegressor(), param, cv=N_SPLIT, refit=True, verbose=1, factor=3, min_resources=kwargs['min_resources'])
        
    st = time.time()
    model.fit(x_train,y_train)
    et = time.time()
    
    y_predict = model.best_estimator_.predict(x_test)
    r2 = r2_score(y_test, y_predict)
    print("R2 : ", r2)
    print("time: ", et-st, "sec")