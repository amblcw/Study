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
from m08_addon import m08_classifier
m08_classifier(x,y)

# r=994
# LOSS: 0.1615818589925766
# ACC:  0.9583371580686616(0.9583371877670288 by loss[1])

# ACC list:  [0.7122, 0.5692, 0.7246, 0.9243, 0.9332, 0.9526]
# Best ML:  RandomForestClassifier

# ACC:  [0.95458809 0.9541492  0.95613673 0.95528476 0.95555154]
# 평균 ACC: 0.9551