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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=123)
param = {'random_state':123}
model_list = [DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor(), XGBRegressor()]

for model in model_list:
    model.fit(x_train,y_train)

    acc = model.score(x_test,y_test)
    print(type(model).__name__,"`s ACC: ",acc,sep='')
    print(type(model).__name__, ":",model.feature_importances_, "\n")

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

# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 4
# min_resources_: 611
# max_resources_: 16512
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 52
# n_resources: 611
# Fitting 5 folds for each of 52 candidates, totalling 260 fits
# ----------
# iter: 1
# n_candidates: 18
# n_resources: 1833
# Fitting 5 folds for each of 18 candidates, totalling 90 fits
# ----------
# iter: 2
# n_candidates: 6
# n_resources: 5499
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# ----------
# iter: 3
# n_candidates: 2
# n_resources: 16497
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# R2 :  0.8037538502833743
# time:  63.721516132354736 sec

# R2 :  0.8012397091272095
# time:  6.444438695907593 sec

# Fitting 5 folds for each of 52 candidates, totalling 260 fits
# ACC:  0.801800516156974

# DecisionTreeRegressor`s ACC: 0.5996278415765189
# DecisionTreeRegressor : [0.51947667 0.04803644 0.04977947 0.02665793 0.03149762 0.13292086
#  0.09825994 0.09337108]

# RandomForestRegressor`s ACC: 0.8151352592204009
# RandomForestRegressor : [0.52465895 0.05290789 0.04857936 0.03009437 0.03183758 0.13291226
#  0.08980219 0.0892074 ]

# GradientBoostingRegressor`s ACC: 0.7978378408140232
# GradientBoostingRegressor : [0.59862955 0.03019079 0.02141612 0.00492335 0.0043039  0.12193619
#  0.10819286 0.11040723]

# XGBRegressor`s ACC: 0.83707103301617
# XGBRegressor : [0.47826383 0.07366086 0.0509511  0.02446287 0.02366972 0.14824368
#  0.0921493  0.10859864]