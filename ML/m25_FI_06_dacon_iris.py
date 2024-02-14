from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import sklearn.preprocessing
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings(action='ignore')

#data
path = "C:\\_data\\DACON\\iris\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submit_csv = pd.read_csv(path+"sample_submission.csv")

x = train_csv.drop(['species'],axis=1)
y = train_csv['species']

print(x,y,sep='\n')
print(x.shape,y.shape)  #x:(120, 4) y:(120,) test_csv:(30, 4)

y = y.to_frame('species')                                      #pandas Series에서 dataframe으로 바꾸는 법

''' 25퍼 미만 열 삭제 '''
# columns = datasets.feature_names
columns = x.columns
x = pd.DataFrame(x,columns=columns)
print("x.shape",x.shape)
''' 이 밑에 숫자에 얻은 feature_importances 넣고 줄 끝마다 \만 붙여주기'''
fi_str = "0.03546066 0.03446177 0.43028001 0.49979756"
 
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

# print(x)
print(f"{x.shape=}, {y.shape=}")      
# print(np.unique(y, return_counts=True)) 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=123)
param = {'random_state':123}
model_list = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

for model in model_list:
    model.fit(x_train,y_train)

    acc = model.score(x_test,y_test)
    print(type(model).__name__,"`s ACC: ",acc,sep='')
    print(type(model).__name__, ":",model.feature_importances_, "\n")
# r=326
# LOSS: 0.3992086946964264
# ACC:  1.0(1.0by loss[1])

# ACC:  [0.95833333 0.95833333 0.95833333 0.875      0.95833333]
# 평균 ACC: 0.9417

# 최적의 매개변수:  RandomForestClassifier(max_depth=6, min_samples_leaf=3)
# 최적의 파라미터:  {'max_depth': 6, 'min_samples_leaf': 3, 'n_estimators': 100}
# best score:  0.9627705627705628
# model score:  0.9166666666666666
# ACC:  0.9166666666666666
# y_pred_best`s ACC: 0.9166666666666666
# time: 3.47sec

# Random
# acc :  0.9583333333333334
# time:  2.3785970211029053 sec

# n_required_iterations: 4
# n_possible_iterations: 2
# min_resources_: 30
# max_resources_: 96
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 52
# n_resources: 30
# Fitting 5 folds for each of 52 candidates, totalling 260 fits
# ----------
# iter: 1
# n_candidates: 18
# n_resources: 90
# Fitting 5 folds for each of 18 candidates, totalling 90 fits
# acc :  0.9583333333333334
# time:  28.62870478630066 sec

# acc :  0.9583333333333334
# time:  0.06525421142578125 sec

# DecisionTreeClassifier`s ACC: 0.9166666666666666
# DecisionTreeClassifier : [0.03546066 0.03446177 0.43028001 0.49979756]

# RandomForestClassifier`s ACC: 0.9583333333333334
# RandomForestClassifier : [0.09657905 0.03354697 0.38571866 0.48415532]

# GradientBoostingClassifier`s ACC: 0.9583333333333334
# GradientBoostingClassifier : [0.0094799  0.0170228  0.41306587 0.56043144]

# XGBClassifier`s ACC: 0.9583333333333334
# XGBClassifier : [0.01676743 0.03789457 0.63928884 0.3060491 ]