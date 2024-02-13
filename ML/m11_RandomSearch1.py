import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# data
x, y = load_iris(return_X_y=True)



from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import time

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=123, stratify=y)

param = [
    {"C":[1,10,100,1000], "kernel":["linear"], "degree":[3,4,5]},                           # 12
    {"C":[1,10,100], "kernel":["rbf"], "gamma":[0.001,0.0001]},                             # 6
    {"C":[1,10,100,1000], "kernel":["sigmoid"],"gamma":[0.01,0.001,0.0001],"degree":[3,4]}  # 24
]

N_SPLIT = 5
kfold = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=123)

# model
model = GridSearchCV(SVC(),param, cv=kfold,verbose=1,refit=True,n_jobs=-1)
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