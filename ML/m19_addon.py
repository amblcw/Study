from sklearn.experimental import enable_halving_search_cv
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings(action='ignore')

def m19_classifier(x,y):
    x_train , x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=333,stratify=y)
    '''
    파이프 라인 안쪽에 파라미터를 넣어줄 때는
    이름+__(2개임)+파라미터 이름
    이렇게 수정해야한다
    '''
    param = [
                {'RF__n_estimators' : [100,200], 'RF__max_depth':[6,10,12], 'RF__min_samples_leaf':[3,10]}, # 12
                {'RF__max_depth':[6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10]},                       # 16
                {'RF__min_samples_leaf': [3,5,7,10], 'RF__min_samples_split':[2,3,5,10]},               # 16
                {'RF__min_samples_split': [2,3,5,10]},                                              # 4
                {'RF__min_samples_split':[2,3,5,10]}                                # 4 총 52
            ]

    pipe = Pipeline([('mm',MinMaxScaler()),('RF',RandomForestClassifier(n_jobs=-1))])

    model = GridSearchCV(pipe, param, cv=5, verbose=1,refit=False)

    model.fit(x_train,y_train)

    loss = model.score(x_test,y_test)
    print("ACC: ",loss)
    
def m19_regressor(x,y):
    x_train , x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=333)
    '''
    파이프 라인 안쪽에 파라미터를 넣어줄 때는
    이름+__(2개임)+파라미터 이름
    이렇게 수정해야한다
    '''
    param = [
                {'RF__n_jobs': [-1],'RF__n_estimators' : [100,200], 'RF__max_depth':[6,10,12], 'RF__min_samples_leaf':[3,10]}, # 12
                {'RF__n_jobs': [-1],'RF__max_depth':[6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10]},                       # 16
                {'RF__n_jobs': [-1],'RF__min_samples_leaf': [3,5,7,10], 'RF__min_samples_split':[2,3,5,10]},               # 16
                {'RF__n_jobs': [-1],'RF__min_samples_split': [2,3,5,10]},                                              # 4
                {'RF__n_jobs': [-1], 'RF__min_samples_split':[2,3,5,10]}                                # 4 총 52
            ]

    pipe = Pipeline([('mm',MinMaxScaler()),('RF',RandomForestRegressor())])

    model = GridSearchCV(pipe, param, cv=5, verbose=1)

    model.fit(x_train,y_train)

    loss = model.score(x_test,y_test)
    print("R2: ",loss)