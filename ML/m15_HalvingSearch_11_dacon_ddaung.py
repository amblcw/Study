from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVR
import warnings
warnings.filterwarnings(action='ignore')

#data
path = "C:\\_data\\DACON\\따릉이\\"
train_csv = pd.read_csv(path+"train.csv",index_col=['id'])  
test_csv = pd.read_csv(path+"test.csv",index_col=0)         
submission_csv = pd.read_csv(path+"submission.csv")

train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'],axis=1) #count 를 드랍, axis=0은 행, axis=1은 열
y = train_csv['count']

from m15_HalvingSearch_00 import m15_regressor
m15_regressor(x,y)

# Epoch 455: early stopping <= best
# loss=1431.286376953125
# r2=0.7554601030711634
# model.add(Dense(512,input_dim=9,activation='relu'))
# model.add(Dense(512,activation='relu'))
# model.add(Dense(256,activation='relu'))
# model.add(Dense(256,activation='relu'))
# model.add(Dense(128,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(1))

# pationce = epo
# loss=1462.5213623046875
# r2=0.7501235084394295

# Epoch 1554: early stopping
# loss=1286.423828125
# r2=0.7802103365358142

# Epoch 2622: early stopping
# loss=1457.3612060546875
# r2=0.75100511942968

# MinMaxScaler
# loss=[1614.5137939453125, 1614.5137939453125]
# r2=0.7241550825071836

# StandardScaler
# loss=[1492.581298828125, 1492.581298828125]
# r2=0.744987662488442

# MaxAbsScaler
# loss=[1119.25048828125, 1119.25048828125]
# r2=0.8087724634833824

# RobustScaler
# loss=[1260.279296875, 1260.279296875]
# r2=0.7846772156960977

# LinearSVR
# loss=0.652966147154205
# r2=0.652966147154205

# ACC list:  [0.493, 0.6613, 0.7235, 0.417, 0.784]
# Best ML:  RandomForestRegressor

# ACC:  [0.7849256  0.76887799 0.74195647 0.78967146 0.78090518]
# 평균 ACC: 0.7733\
    
# 최적의 매개변수:  RandomForestRegressor(n_jobs=4)
# 최적의 파라미터:  {'min_samples_split': 2, 'n_jobs': 4}
# best score:  0.7738920598415189
# model score:  0.8189193246824287
# R2:  0.8189193246824287
# y_pred_best`s R2: 0.8189193246824287
# time: 6.05sec

# Random
# r2 :  0.7973734338387666
# time:  3.2512457370758057 sec

# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 4
# min_resources_: 43
# max_resources_: 1167
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 52
# n_resources: 43
# Fitting 5 folds for each of 52 candidates, totalling 260 fits
# ----------
# iter: 1
# n_candidates: 18
# n_resources: 129
# Fitting 5 folds for each of 18 candidates, totalling 90 fits
# ----------
# iter: 2
# n_candidates: 6
# n_resources: 387
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# ----------
# iter: 3
# n_candidates: 2
# n_resources: 1161
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# R2 :  0.794429063481833
# time:  31.23870849609375 sec