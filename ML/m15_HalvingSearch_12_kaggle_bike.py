from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import time
import math
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVR
import warnings
warnings.filterwarnings(action='ignore')

#data
path = "C:\\_data\\KAGGLE\\bike-sharing-demand\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submission_csv = pd.read_csv(path+"sampleSubmission.csv")

x = train_csv.drop(['casual','registered','count'],axis=1)
y = train_csv['count']

print(x.shape, y.shape)

from m15_HalvingSearch_00 import m15_regressor
m15_regressor(x,y)

# LinearSVR
# RMSLE:  1.256727136597206

# ACC list:  [-0.096, 0.0603, 0.0309, -0.1751, 0.0938]
# Best ML:  RandomForestRegressor

# ACC:  [0.27476635 0.29534065 0.28111063 0.32957913 0.31385862]
# 평균 ACC: 0.2989

# (10886, 8) (10886,)
# Fitting 5 folds for each of 60 candidates, totalling 300 fits
# 최적의 매개변수:  RandomForestRegressor(max_depth=10, min_samples_leaf=3, n_estimators=200)
# 최적의 파라미터:  {'max_depth': 10, 'min_samples_leaf': 3, 'n_estimators': 200}
# best score:  0.3612533291192763
# model score:  0.3267798707073968
# R2:  0.3267798707073968
# y_pred_best`s R2: 0.3267798707073968
# time: 15.71sec

# Random
# r2 :  0.3546122285399441
# time:  4.877974510192871 sec

# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 4
# min_resources_: 322
# max_resources_: 8708
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 52
# n_resources: 322
# Fitting 5 folds for each of 52 candidates, totalling 260 fits
# ----------
# iter: 1
# n_candidates: 18
# n_resources: 966
# Fitting 5 folds for each of 18 candidates, totalling 90 fits
# ----------
# iter: 2
# n_candidates: 6
# n_resources: 2898
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# ----------
# iter: 3
# n_candidates: 2
# n_resources: 8694
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# R2 :  0.3560828034800847
# time:  32.55228042602539 sec