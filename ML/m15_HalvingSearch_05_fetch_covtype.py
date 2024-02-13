from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings(action='ignore')


datasets = fetch_covtype()
x = datasets.data  
y = datasets.target

x = x[:100000]
y = y[:100000]
# print(x.shape,y.shape)      #(581012, 54) (581012,)
# print(pd.value_counts(y))
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747

# y = pd.get_dummies(y)
# ohe = OneHotEncoder(sparse=False)
# y = ohe.fit_transform(y.reshape(-1,1))

# print(y,y.shape,sep='\n')
# print(np.count_nonzero(y[:,0]))
'''
sklearn : (581012, 7)
pandas  : (581012, 7)
keras   : (581012, 8)
keras 첫번째 열이 미심직어 찍어보니
print(np.count_nonzero(y[:,0])) # 0
따라서 첫번째 열 잘라내고 슬라이싱
'''
# print(y.shape)

# print(y,y.shape,sep='\n')       # (581012, 7)
# print(np.count_nonzero(y[:,0])) # 211840
from m15_HalvingSearch_00 import m15_classifier
m15_classifier(x,y)
# r=994
# LOSS: 0.1615818589925766
# ACC:  0.9583371580686616(0.9583371877670288 by loss[1])

# ACC list:  [0.7122, 0.5692, 0.7246, 0.9243, 0.9332, 0.9526]
# Best ML:  RandomForestClassifier

# ACC:  [0.95458809 0.9541492  0.95613673 0.95528476 0.95555154]
# 평균 ACC: 0.9551

# 최적의 매개변수:  RandomForestClassifier(n_jobs=2)
# 최적의 파라미터:  {'min_samples_split': 2, 'n_jobs': 2}
# best score:  0.9430888888888889
# model score:  0.9469
# ACC:  0.9469
# y_pred_best`s ACC: 0.9469
# time: 296.03sec

# Random
# acc :  0.9256
# time:  34.10847043991089 sec

# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 4
# min_resources_: 2962
# max_resources_: 80000
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 52
# n_resources: 2962
# Fitting 5 folds for each of 52 candidates, totalling 260 fits
# ----------
# iter: 1
# n_candidates: 18
# n_resources: 8886
# Fitting 5 folds for each of 18 candidates, totalling 90 fits
# ----------
# iter: 2
# n_candidates: 6
# n_resources: 26658
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# ----------
# iter: 3
# n_candidates: 2
# n_resources: 79974
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# acc :  0.9454
# time:  82.62311005592346 sec