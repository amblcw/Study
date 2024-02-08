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

from m10_addon import m10
m10(x,y,'r')

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