from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
import time

datasets = load_digits()
x = datasets.data   
y = datasets.target

print(x.shape, y.shape) # (1797, 64) (1797,)
print(np.unique(y,return_counts=True))  # 다중분류 확인
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64)) 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=123, stratify=y)

param = [
        {'n_jobs': [-1],'n_estimators' : [100,200], 'max_depth':[6,10,12], 'min_samples_leaf':[3,10]}, # 12
        {'n_jobs': [-1],'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10]},                       # 16
        {'n_jobs': [-1],'min_samples_leaf': [3,5,7,10], 'min_samples_split':[2,3,5,10]},               # 16
        {'n_jobs': [-1],'min_samples_split': [2,3,5,10]},                                              # 4
        {'n_jobs': [-1], 'min_samples_split':[2,3,5,10]}                                # 4 총 52
    ]

N_SPLIT = 5
k_fold = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=333)

# model = RandomForestClassifier()
# model = GridSearchCV(RandomForestClassifier(), param, cv= k_fold, refit=True, n_jobs=-1)
model = RandomizedSearchCV(RandomForestClassifier(),param,cv=k_fold,refit=True,n_jobs=-1)

st = time.time()
model.fit(x_train,y_train)
et = time.time()

loss = model.score(x_test,y_test)
print("acc: ",loss)
print(et-st,"sec")

# default
# acc:  1.0
# 0.30097126960754395 sec

# GridSearchCV
# acc:  1.0
# 5.885230302810669 sec

# RandomizedSearchCV
# acc:  1.0
# 2.4038641452789307 sec