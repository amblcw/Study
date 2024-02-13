import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# data
# x, y = load_iris(return_X_y=True)
x, y = load_digits(return_X_y=True)

print(np.unique(y,return_counts=True))

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import accuracy_score
import time

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123, stratify=y)

param = [
    {"C":[1,10,100,1000], "kernel":["linear"], "degree":[3,4,5]},                           # 12
    {"C":[1,10,100], "kernel":["rbf"], "gamma":[0.001,0.0001]},                             # 6
    {"C":[1,10,100,1000], "kernel":["sigmoid"],"gamma":[0.01,0.001,0.0001],"degree":[3,4]}  # 24
]

N_SPLIT = 5
kfold = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=123)

# model
model = HalvingGridSearchCV(SVC(),param, cv=kfold,verbose=1,refit=True,n_jobs=-1, factor=3, min_resources=159)
# model = RandomizedSearchCV(SVC(),param, cv=kfold,verbose=1,refit=True,n_jobs=-1)

st = time.time()
model.fit(x_train,y_train)
et = time.time()
print("최적의 매개변수: ", model.best_estimator_)
print("최적의 파라미터: ", model.best_params_)
print("best score: ", model.best_score_)    # train 과정에서 최고성적
print("model score: ", model.score(x_test,y_test))

# 최적의 매개변수:  SVC(C=1, kernel='linear')                 # best_params_에서 찾은 값들로 다시 한번 조합을 찾아본 최적의 값
# 최적의 파라미터:  {'C': 1, 'degree': 3, 'kernel': 'linear'} # 우리가 지정한 값중 제일 좋은 값을 뽑는다
# best score:  0.9925925925925926
# model score:  0.9333333333333333

y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)
print("ACC: ",acc)

y_pred_best = model.best_estimator_.predict(x_test)
print("y_pred_best`s ACC:", accuracy_score(y_test,y_pred_best))

print(f"time: {et-st:.2f}sec")

# print(pd.DataFrame(model.cv_results_))

# Grid
# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# 최적의 매개변수:  SVC(C=1, kernel='linear')
# 최적의 파라미터:  {'C': 1, 'degree': 3, 'kernel': 'linear'}
# best score:  0.9925925925925926
# model score:  0.9333333333333333
# ACC:  0.9333333333333333
# y_pred_best`s ACC: 0.9333333333333333
# time: 1.25sec

# Random
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수:  SVC(C=10, kernel='linear')
# 최적의 파라미터:  {'kernel': 'linear', 'degree': 3, 'C': 10}
# best score:  0.9777777777777779
# model score:  0.8666666666666667
# ACC:  0.8666666666666667
# y_pred_best`s ACC: 0.8666666666666667
# time: 1.26sec


'''
매 단계마다 파라미터 조합은 factor만큼 나누어 줄어들고 데이터는 factor배 만큼 증가한다
시작 데이터 개수 = CV * 2 * label + alpha 의 개수 (라고 되어는 있지만 알고리즘에 의해 조금씩 틀어짐)(alpha는 내부 알고리즘에 의한 조정값)
'''
# n_iterations: 3
# n_required_iterations: 3
# n_possible_iterations: 3
# min_resources_: 89
# max_resources_: 1437
# aggressive_elimination: False
# factor: 4
# ----------
# iter: 0
# n_candidates: 42
# n_resources: 89
# Fitting 3 folds for each of 42 candidates, totalling 126 fits
# ----------
# iter: 1
# n_candidates: 11
# n_resources: 356
# Fitting 3 folds for each of 11 candidates, totalling 33 fits
# ----------
# iter: 2
# n_candidates: 3
# n_resources: 1424
# Fitting 3 folds for each of 3 candidates, totalling 9 fits
# 최적의 매개변수:  SVC(C=1, degree=5, kernel='linear')
# 최적의 파라미터:  {'C': 1, 'degree': 5, 'kernel': 'linear'}
# best score:  0.9760900140646975
# model score:  0.9861111111111112
# ACC:  0.9861111111111112
# y_pred_best`s ACC: 0.9861111111111112
# time: 1.50sec