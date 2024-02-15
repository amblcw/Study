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
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

datasets = load_boston()
x = datasets.data
y = datasets.target

r2 = 0

''' 25퍼 미만 열 삭제 '''
columns = datasets.feature_names
# columns = x.columns
x = pd.DataFrame(x,columns=columns)
print("x.shape",x.shape)
''' 이 밑에 숫자에 얻은 feature_importances 넣고 줄 끝마다 \만 붙여주기'''
fi_str = "0.03975657 0.00070606 0.00619758 0.00133015 0.0214521  0.61806582\
 0.00812987 0.05990651 0.00313737 0.01269828 0.01219055 0.00519019\
 0.21123894"
 
''' str에서 숫자로 변환하는 구간 '''
fi_str = fi_str.split()
fi_float = [float(s) for s in fi_str]
print(fi_float)
fi_list = pd.Series(fi_float)

''' 25퍼 미만 인덱스 구하기 '''
low_idx_list = fi_list[fi_list <= fi_list.quantile(0.25)].index
print('low_idx_list',low_idx_list)

''' 25퍼 미만 제거하기 '''
low_col_list = [x.columns[index] for index in low_idx_list]
# 이건 혹여 중복되는 값들이 많아 25퍼이상으로 넘어갈시 25퍼로 자르기
if len(low_col_list) > len(x.columns) * 0.25:   
    low_col_list = low_col_list[:int(len(x.columns)*0.25)]
print('low_col_list',low_col_list)
x.drop(low_col_list,axis=1,inplace=True)
print("after x.shape",x.shape)

from sklearn.decomposition import PCA 
origin_x = x
print('x.shape',x.shape)
for i in range(1,x.shape[1]):
    pca = PCA(n_components=i)
    x = pca.fit_transform(origin_x)

    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=123)
    param = {'random_state':123}
    model_list = [RandomForestRegressor()]

    for model in model_list:
        model.fit(x_train,y_train)

        acc = model.score(x_test,y_test)
        print(type(model).__name__,"`s ACC: ",acc,sep='')
        print(type(model).__name__, ":",model.feature_importances_, "\n")

EVR = pca.explained_variance_ratio_
print(EVR)
print(np.cumsum(EVR))
# [8.15877164e-01 1.65719172e-01 1.58082868e-02 1.29214254e-03
#  7.43280970e-04 4.18597843e-04 9.46875544e-05 3.95029913e-05
#  7.08249727e-06]
# [0.81587716 0.98159634 0.99740462 0.99869677 0.99944005 0.99985864
#  0.99995333 0.99999283 0.99999992]

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

# after
# DecisionTreeRegressor`s ACC: 0.4492743117079012
# DecisionTreeRegressor : [0.0406138  0.00472457 0.02330789 0.62058219 0.00622547 0.0616007
#  0.01450596 0.01146437 0.00655538 0.21041969]

# RandomForestRegressor`s ACC: 0.7729849169407195
# RandomForestRegressor : [0.02920261 0.00548395 0.02237045 0.51650164 0.01455211 0.04108818
#  0.01590827 0.01384595 0.00929459 0.33175225]

# GradientBoostingRegressor`s ACC: 0.8104832534527571
# GradientBoostingRegressor : [0.0159612  0.00180914 0.03178777 0.48946006 0.01271472 0.0587632
#  0.01298416 0.02444633 0.00609386 0.34597958]

# XGBRegressor`s ACC: 0.8037710183512754
# XGBRegressor : [0.01644142 0.00784428 0.03727875 0.49134448 0.02142194 0.04706337
#  0.03374919 0.03786127 0.01057633 0.29641888]

# pca
# XGBRegressor`s ACC: 0.825738339781916
# XGBRegressor : [0.10840572 0.01494553 0.05480283 0.07863774 0.38822186 0.04070629
#  0.06259476 0.01991601 0.23176931]