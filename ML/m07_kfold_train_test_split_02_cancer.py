import numpy as np
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import LinearSVC

import warnings
warnings.filterwarnings(action='ignore')

# data
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)
x = datasets.data
y = datasets.target 
df_y = pd.DataFrame(y)

# print(df_y)
# print(x,y,x.shape,y.shape,sep='\n')
print(np.unique(y,return_counts=True)) #(array([0, 1]), array([212, 357], dtype=int64))
zero_num = len(y[np.where(y == 0)]) #y[np.where(조건)]은 조건에 맞는 값들의 인덱스 리스트를 반환
one_num = len(y[np.where(y == 1)])
print(f"0: {zero_num}, 1: {one_num}")
print(df_y.value_counts()) #pandas 요소 개수 세기
print(pd.value_counts(y))  #위와 동일

from m07_addon import m07_classifier
m07_classifier(x,y)

# 128~1(step:1/2), all relu, No shuffle
# Epoch 1018: early stopping
# 6/6 [==============================] - 0s 603us/step
# ACCURACY: 0.9766082167625427

# Best result : SVC`s 0.9825

# ACC:  [0.92982456 0.92105263 0.92982456 0.92982456 0.87610619]
# 평균 ACC: 0.9173
# 평균 ACC: 0.9191