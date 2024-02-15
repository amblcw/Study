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

''' 25퍼 미만 열 삭제 '''
# columns = datasets.feature_names
columns = x.columns
x = pd.DataFrame(x,columns=columns)
print("x.shape",x.shape)
''' 이 밑에 숫자에 얻은 feature_importances 넣고 줄 끝마다 \만 붙여주기'''
fi_str = "0.58096191 0.17320882 0.02719788 0.01812347 0.03072869 0.05335738\
 0.04302835 0.04255812 0.03083538"
 
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
# [9.96279502e-01 2.62175571e-03 6.22370491e-04 3.02112120e-04
#  1.25702570e-04 4.85567784e-05]
# [0.9962795  0.99890126 0.99952363 0.99982574 0.99995144 1.        ]

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

# R2 :  0.7991863783426085
# time:  0.2980027198791504 sec

# Fitting 5 folds for each of 52 candidates, totalling 260 fits
# R2:  0.79859624573135

# DecisionTreeRegressor`s ACC: 0.6225334968278757
# DecisionTreeRegressor : [0.58096191 0.17320882 0.02719788 0.01812347 0.03072869 0.05335738
#  0.04302835 0.04255812 0.03083538]

# RandomForestRegressor`s ACC: 0.7855401405350986
# RandomForestRegressor : [0.57978664 0.1862202  0.02131647 0.03203421 0.03550776 0.04354021
#  0.04233916 0.03501987 0.02423547]

# GradientBoostingRegressor`s ACC: 0.7548235948778885
# GradientBoostingRegressor : [0.63573962 0.21408034 0.02440254 0.01607164 0.01146122 0.03729284
#  0.02738591 0.02131794 0.01224797]

# XGBRegressor`s ACC: 0.7632837109221342
# XGBRegressor : [0.34626618 0.09873444 0.36363754 0.01909325 0.02808662 0.04098203
#  0.04632215 0.03015554 0.02672229]

# after
# DecisionTreeRegressor`s ACC: 0.6038129282026836
# DecisionTreeRegressor : [0.58410787 0.18609834 0.04564533 0.05353262 0.0514235  0.05039775
#  0.02879459]

# RandomForestRegressor`s ACC: 0.7967212630932444
# RandomForestRegressor : [0.59225627 0.19045739 0.04652447 0.05288092 0.04990394 0.04043135
#  0.02754568]

# GradientBoostingRegressor`s ACC: 0.7551113507739712
# GradientBoostingRegressor : [0.64719976 0.21400846 0.02499513 0.04166271 0.03834928 0.02191041
#  0.01187424]

# XGBRegressor`s ACC: 0.7701087884483183
# XGBRegressor : [0.54293376 0.16813059 0.05376956 0.06909224 0.06736376 0.05250703
#  0.04620307]