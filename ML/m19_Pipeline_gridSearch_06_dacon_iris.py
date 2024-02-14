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


# print(x)
print(f"{x.shape=}, {y.shape=}")      
# print(np.unique(y, return_counts=True)) 
from m19_addon import m19_classifier
m19_classifier(x,y)
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