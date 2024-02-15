import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pandas as pd
import optuna
import time

import warnings
warnings.filterwarnings('ignore')
'''
tree_mothod='gpu_hist',
predictor='gou_predictor',
gpu_id=0
'''
# data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")
# x_train.shape=(60000, 28, 28)
# x_test.shape=(10000, 28, 28)
# y_train.shape=(60000,)
# y_test.shape=(10000,)

# x = np.append(x_train,x_test, axis=0)
# x = np.concatenate([x_train,x_test], axis=0)
x = np.vstack([x_train,x_test])
print(x.shape)  # (70000, 28, 28)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components=x.shape[1])
x1 = pca.fit_transform(x)
EVR = pca.explained_variance_ratio_
EVR_sum = np.cumsum(EVR)
evr_sum = pd.Series(EVR_sum).round(decimals=6)
print(evr_sum)
print(len(evr_sum[evr_sum >= 0.95]))
print(len(evr_sum[evr_sum >= 0.99]))
print(len(evr_sum[evr_sum >= 0.999]))
print(len(evr_sum[evr_sum >= 1.0]))
print("0.95  커트라인 n_components: ",len(evr_sum[evr_sum < 0.95]))
print("0.99  커트라인 n_components: ",len(evr_sum[evr_sum < 0.99]))
print("0.999 커트라인 n_components: ",len(evr_sum[evr_sum < 0.999]))
print("1.0   커트라인 n_components: ",len(evr_sum[evr_sum < 1.0]))
print(evr_sum.iloc[331])    # 0.950031
print(evr_sum.iloc[543])    # 0.990077
print(evr_sum.iloc[682])    # 0.999023
print(evr_sum.iloc[712])    # 1.0

cutline = [
    (len(evr_sum[evr_sum < 0.95]), round(evr_sum.iloc[331],4)),
    (len(evr_sum[evr_sum < 0.99]), round(evr_sum.iloc[543],4)),
    (len(evr_sum[evr_sum < 0.999]), round(evr_sum.iloc[682],4)),
    (len(evr_sum[evr_sum < 1.0]), round(evr_sum.iloc[712],4)),
    (784, '전체 데이터')
]



import matplotlib.pyplot as plt
plt.plot(evr_sum)
plt.grid()
# plt.show()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

pca = PCA(n_components=x_train.shape[1]-1).fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)


y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

param = [
        {'n_estimators' : [50,70,100], 'max_depth':[2,3,6], 
         'learning_rate':[0.01,0.03,0.05],
        #  'tree_method':['hist'],
        #  'predictor': ['gpu_predictor'],
        #  'device': ['cuda'],
         }, # 12
    ]

def objectiveXGB(trial):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 5, 40),
        'max_depth' : trial.suggest_int('max_depth', 8, 16),
        # 'min_child_weight' : trial.suggest_int('min_child_weight', 1, 300),
        'gamma' : trial.suggest_int('gamma', 1, 3),
        'learning_rate' : trial.suggest_loguniform('learning_rate',0.01,0.1),
        # 'colsample_bytree' : trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1, 0.1),
        # 'nthread' : -1,
        'tree_method' : 'hist',
        'device' : 'cuda',
        # 'lambda' : trial.suggest_loguniform('lambda', 1e-3, 10.0),
        # 'alpha' : trial.suggest_loguniform('alpha', 1e-3, 10.0),
        # 'subsample' : trial.suggest_categorical('subsample', [0.6,0.7,0.8,1.0]),
        # 'random_state' : trial.suggest_int('random_state', 1, 1000)
    }
    # 학습 모델 생성
    model = XGBClassifier(**param)
    xgb_model = model.fit(x_train, y_train) # 학습 진행
    
    # 모델 성능 확인
    score = accuracy_score(xgb_model.predict(x_test), y_test)
    
    return score

acc_list = []
for cut_idx, cut_num in cutline:
    pca = PCA(n_components=cut_idx)
    pca_x = pca.fit_transform(x)
    
    x_train = pca_x[:60000]
    x_test = pca_x[60000:]
    print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")
    # model
    model = XGBClassifier(tree_method='hist',device='cuda')
    # model = RandomizedSearchCV(XGBClassifier(tree_method='hist',device='cuda'),param, cv=3, refit=True, n_jobs=-1, n_iter=5, random_state=32, verbose=1)
    
    # compile & fit
    study = optuna.create_study(direction='maximize')
    study.optimize(objectiveXGB, n_trials=300)

    best_params = study.best_params
    print(best_params)
    
    model = XGBClassifier(best_params)
    
    start_time = time.time()
    model.fit(x_train,y_train)
    end_time = time.time()
    # evaluate & predict
    y_predict = model.predict(x_test)
    # print(model.best_params_)
    # y_predict = model.best_estimator_.predict(x_test)
    acc = accuracy_score(y_test,y_predict)
    
    print(f"time: {end_time - start_time}sec")
    print(f"{cut_num}의 ACC:  {acc}\n\n") 
    acc_list.append((round(end_time - start_time,4), cut_num,round(acc,4)))
    
for time, c_n , acc in acc_list:
    print(f"{c_n:<6}의 ACC:  {acc}") 
    print(f"time: {time}sec")
    
""" 
0.95  의 ACC:  0.9747
time: 63.7506sec
0.9901의 ACC:  0.9716
time: 64.8889sec
0.999 의 ACC:  0.9696
time: 65.8921sec
1.0   의 ACC:  0.9674
time: 66.8954sec
전체 데이터의 ACC:  0.9701
time: 66.0671sec
"""
""" 
Randomnizer의 xgboost
0.95  의 ACC:  0.8771
time: 77.3356sec
0.9901의 ACC:  0.8741
time: 112.8319sec
0.999 의 ACC:  0.8753
time: 199.2867sec
1.0   의 ACC:  0.8766
time: 232.7666sec
전체 데이터의 ACC:  0.8765
time: 202.4123sec """
""" 
xgboost deault
0.95  의 ACC:  0.9244
time: 8.4701sec
0.9901의 ACC:  0.9223
time: 12.5756sec
0.999 의 ACC:  0.9201
time: 15.4817sec
1.0   의 ACC:  0.9204
time: 15.7807sec
전체 데이터의 ACC:  0.9185
time: 17.199sec """