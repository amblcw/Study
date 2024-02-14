from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_wine
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

datasets = load_wine()
x = datasets.data
y = datasets.target

# print(x.shape,y.shape)  #(178, 13) (178,)
# print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48
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
# r=398
# LOSS: 0.13753800094127655
# ACC:  1.0(1.0 by loss[1])

# Best result : SVC`s 1.0000

# ACC:  [0.72222222 0.72222222 0.61111111 0.62857143 0.74285714]
# 평균 ACC: 0.6854

# 최적의 매개변수:  RandomForestClassifier(max_depth=6, min_samples_leaf=3)
# 최적의 파라미터:  {'max_depth': 6, 'min_samples_leaf': 3}
# best score:  0.9875
# model score:  1.0
# ACC:  1.0
# y_pred_best`s ACC: 1.0
# time: 2.86sec

# Random
# acc :  0.7251908396946565
# time:  2.465505361557007 sec

# n_iterations: 2
# n_required_iterations: 4
# n_possible_iterations: 2
# min_resources_: 30
# max_resources_: 142
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
# acc :  0.9722222222222222
# time:  26.515902996063232 sec

# acc :  0.9722222222222222
# time:  0.06301736831665039 sec

# DecisionTreeClassifier`s ACC: 0.9444444444444444
# DecisionTreeClassifier : [0.0420676  0.05312189 0.         0.         0.03888017 0.
#  0.12962152 0.         0.         0.         0.02912372 0.27920685
#  0.42797824]

# RandomForestClassifier`s ACC: 1.0
# RandomForestClassifier : [0.13654918 0.04281305 0.01488527 0.01598451 0.03649194 0.05446449
#  0.15260314 0.00880178 0.02256245 0.12307229 0.07427409 0.11521899
#  0.20227884]

# GradientBoostingClassifier`s ACC: 0.9722222222222222
# GradientBoostingClassifier : [0.03398654 0.0781348  0.00310596 0.00156517 0.0125634  0.00456263
#  0.08202304 0.00248588 0.00082658 0.23676842 0.06381941 0.16026778
#  0.3198904 ]

# XGBClassifier`s ACC: 1.0
# XGBClassifier : [0.0363352  0.05855075 0.0201256  0.00602301 0.02279056 0.01907869
#  0.06725097 0.0003421  0.02408856 0.14139089 0.13685898 0.3411003
#  0.12606439]