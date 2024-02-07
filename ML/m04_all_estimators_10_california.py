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
r2=0
# while r2 < 0.6: 
r = int(np.random.uniform(1,1000))
r = 176
# r = 130
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,random_state=r)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler().fit(x_train)    #최솟값을 0 최댓값을 1로 스케일링
# scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
scaler = RobustScaler().fit(x_train)    #

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#model
from sklearn.utils import all_estimators

# all_algorithms = all_estimators(type_filter='classifier')
all_algorithms = all_estimators(type_filter='regressor')
# print(len(all_algorithms))  # 41(분류) 55(회귀) 
result_list = []
error_list = []
for name, algorithm in all_algorithms:
    try:
        model = algorithm()
        model.fit(x_train,y_train)
        acc = model.score(x_test,y_test)
    except Exception as e:
        print(f"{name:30} ERROR")
        error_list.append(e)
        continue
    print(f"{name:30} ACC: {acc:.4f}")
    result_list.append((name,acc))
    
# print('error_list: \n',error_list)
best_result = max(result_list)[1]
best_algirithm = result_list[result_list.index(max(result_list))][0]
print(f'\nBest result : {best_algirithm}`s {best_result:.4f}')

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