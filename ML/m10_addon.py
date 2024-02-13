import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score
import time

def m10(x,y,c_r):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=123)

    param = [
        {'n_jobs': [-1],'n_estimators' : [100,200], 'max_depth':[6,10,12], 'min_samples_leaf':[3,10]}, # 12
        {'n_jobs': [-1],'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10]},                       # 16
        {'n_jobs': [-1],'min_samples_leaf': [3,5,7,10], 'min_samples_split':[2,3,5,10]},               # 16
        {'n_jobs': [-1],'min_samples_split': [2,3,5,10]},                                              # 4
        {'n_jobs': [-1], 'min_samples_split':[2,3,5,10]}                                # 4 총 52
    ]

    N_SPLIT = 5
    kfold = KFold(n_splits=N_SPLIT, shuffle=True, random_state=123)

    # model
    model = 0
    if c_r == "c":  
        model = GridSearchCV(RandomForestClassifier(),param, cv=kfold,verbose=1,refit=True,n_jobs=-1)
    if c_r == "r":
        model = GridSearchCV(RandomForestRegressor(),param, cv=kfold,verbose=1,refit=True,n_jobs=-1)

    st = time.time()
    model.fit(x_train,y_train)
    et = time.time()
    print("최적의 매개변수: ", model.best_estimator_)
    print("최적의 파라미터: ", model.best_params_)
    print("best score: ", model.best_score_)    # train 과정에서 최고성적
    print("model score: ", model.score(x_test,y_test))

    # 최적의 매개변수:  SVC(C=1, kernel='linear')                 # best_params_에서 찾은 값들로 다시 한번 조합을 찾아본 최적의 값
    # 최적의 파라미터:  {'C': 1, 'degree': 3, 'kernel': 'linear'} # 우리가 지정한 값중 제일 좋은 값을 뽑는다
    # best score:  0.9925925925925926
    # model score:  0.9333333333333333

    y_predict = model.predict(x_test)
    y_pred_best = model.best_estimator_.predict(x_test)
    
    if c_r=='c':
        print("ACC: ",accuracy_score(y_test,y_predict))
        print("y_pred_best`s ACC:", accuracy_score(y_test,y_pred_best))
    if c_r=='r':
        print("R2: ",r2_score(y_test,y_predict))
        print("y_pred_best`s R2:", r2_score(y_test,y_pred_best))
        

    print(f"time: {et-st:.2f}sec")

    # print(pd.DataFrame(model.cv_results_))
