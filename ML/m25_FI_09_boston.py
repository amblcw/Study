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

# DecisionTreeRegressor`s ACC: 0.4223909446145995
# DecisionTreeRegressor : [0.03975657 0.00070606 0.00619758 0.00133015 0.0214521  0.61806582
#  0.00812987 0.05990651 0.00313737 0.01269828 0.01219055 0.00519019
#  0.21123894]

# RandomForestRegressor`s ACC: 0.7765389639546698
# RandomForestRegressor : [0.03011428 0.00104815 0.0049405  0.00104661 0.01957681 0.55534768
#  0.01235241 0.04757514 0.00357361 0.01603252 0.01172268 0.00922618
#  0.28744344]

# GradientBoostingRegressor`s ACC: 0.8125451312716561
# GradientBoostingRegressor : [1.59981256e-02 3.89067701e-04 1.24966647e-03 4.95965556e-05
#  3.13141443e-02 4.88782448e-01 1.30130003e-02 5.91280739e-02
#  1.35607957e-03 1.16908589e-02 2.46070702e-02 5.65393590e-03
#  3.46767932e-01]

# XGBRegressor`s ACC: 0.8031678716317369
# XGBRegressor : [0.01676424 0.00214259 0.01079342 0.00273458 0.04193499 0.4363328
#  0.02148097 0.05215991 0.01775352 0.04331461 0.03744346 0.01197497
#  0.30516994]