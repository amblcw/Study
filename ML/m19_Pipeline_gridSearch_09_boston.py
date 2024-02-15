from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import load_boston    # pip install scikit-learn==1.1.3
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.svm import LinearSVR

import warnings
warnings.filterwarnings('ignore')

datasets = load_boston()
x = datasets.data
y = datasets.target

r2 = 0

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

    model = make_pipeline(MinMaxScaler(),RandomForestRegressor(**param))
    
    st = time.time()
    model.fit(x_train,y_train)
    et = time.time()
    
    r2 = model.score(x_test, y_test)
    print("R2 : ", r2)
    print("time: ", et-st, "sec")
    
param = {'max_depth': 12, 'min_samples_leaf': 3}
m15_classifier(x, y, param)

# StandardScaler
# loss=[4.916057586669922, 1.7392468452453613]
# r2=0.9333945146886241
# RMSE: 2.217218335403849

# ACC:  [0.75567667 0.86588395 0.9117018  0.92818409 0.86901632]
# 평균 ACC: 0.8661

# 최적의 매개변수:  RandomForestRegressor(min_samples_split=5, n_jobs=2)
# 최적의 파라미터:  {'min_samples_split': 5, 'n_jobs': 2}
# best score:  0.8583454484421443
# model score:  0.7032622274103404
# R2:  0.7032622274103404
# y_pred_best`s R2: 0.7032622274103404
# time: 3.65sec