from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVR
import warnings
warnings.filterwarnings(action='ignore')

#data
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape,y.shape,sep='\n')
print(datasets.feature_names)   
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
from m13_RandomSearch_00 import m13_regressor
m13_regressor(x,y)

# Epoch 236: early stopping
# Time: 35.14sec
# r=176
# loss=0.45041927695274353
# r2=0.661553367308052

# MinMaxScaler
# loss=[0.27136003971099854, 0.27136003971099854]
# r2=0.796099072829003

# StandardScaler
# loss=[0.26987117528915405, 0.26987117528915405]
# r2=0.7972178874447766

# MaxAbsScaler
# loss=[0.3393670916557312, 0.3393670916557312]
# r2=0.7449983684198338

# RobustScaler
# loss=[0.293599396944046, 0.293599396944046]
# r2=0.7793883660319727

# LinearSVR
# r=176
# loss=0.5744965050202392
# r2=0.5744965050202392

# ACC list:  [0.6786, 0.6075, 0.6904, 0.6227, 0.8194]
# Best ML:  RandomForestRegressor

# ACC:  [0.81405683 0.82544473 0.81082978 0.79367265 0.80587449]
# 평균 ACC: 0.81

# 최적의 매개변수:  RandomForestRegressor(min_samples_split=3, n_jobs=4)
# 최적의 파라미터:  {'min_samples_split': 3, 'n_jobs': 4}
# best score:  0.8065723786562502
# model score:  0.8205521057965446
# R2:  0.8205521057965446
# y_pred_best`s R2: 0.8205521057965446
# time: 76.53sec

# Random
# r2 :  0.8016798072818568
# time:  39.16767120361328 sec