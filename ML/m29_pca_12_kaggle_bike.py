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

''' 25퍼 미만 열 삭제 '''
# columns = datasets.feature_names
columns = x.columns
x = pd.DataFrame(x,columns=columns)
print("x.shape",x.shape)
''' 이 밑에 숫자에 얻은 feature_importances 넣고 줄 끝마다 \만 붙여주기'''
fi_str = "0.06767629 0.00623194 0.04226798 0.04526605 0.1310375  0.24831268\
 0.25612224 0.20308532"
 
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
# [0.66356618 0.23031791 0.10198196 0.00192793 0.00163736]
# [0.66356618 0.89388409 0.99586605 0.99779397 0.99943133]

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

# R2 :  0.3669200796299886
# time:  1.7153191566467285 sec

# Fitting 5 folds for each of 52 candidates, totalling 260 fits
# R2:  0.36535160161988023

# DecisionTreeRegressor`s ACC: -0.13683855108363274
# DecisionTreeRegressor : [0.06767629 0.00623194 0.04226798 0.04526605 0.1310375  0.24831268
#  0.25612224 0.20308532]

# RandomForestRegressor`s ACC: 0.2683721206884162
# RandomForestRegressor : [0.06808358 0.00607502 0.04257098 0.052278   0.14164557 0.23940337
#  0.25187026 0.19807324]

# GradientBoostingRegressor`s ACC: 0.3151204338624217
# GradientBoostingRegressor : [0.0825438  0.00096736 0.03573283 0.01654246 0.20167153 0.31201157
#  0.33182993 0.01870051]

# XGBRegressor`s ACC: 0.2979634934672556
# XGBRegressor : [0.12239873 0.04252511 0.10369455 0.07675308 0.10646653 0.33821073
#  0.14905396 0.06089731]

# after
# DecisionTreeRegressor`s ACC: -0.16076384979035274
# DecisionTreeRegressor : [0.0720547  0.052627   0.13802408 0.24912474 0.26100102 0.22716845]

# RandomForestRegressor`s ACC: 0.2383020948789052
# RandomForestRegressor : [0.07410588 0.05326697 0.1474628  0.24514613 0.26586265 0.21415557]

# GradientBoostingRegressor`s ACC: 0.3030279534078809
# GradientBoostingRegressor : [0.0850965  0.01764199 0.20338207 0.32982783 0.34194393 0.02210767]

# XGBRegressor`s ACC: 0.27455842263333274
# XGBRegressor : [0.1554413  0.08725202 0.14018974 0.36397374 0.18003702 0.07310625]