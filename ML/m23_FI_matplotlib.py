import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

print(sklearn.__version__)

# data
datasets = load_iris()
x = datasets.data
y = datasets.target

def plot_FI(model):
    n_features = x.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1,n_features)
    plt.title(type(model).__name__)
    plt.show()

print(np.unique(y,return_counts=True))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123, stratify=y)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

param = {'random_state':123}
model_list = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

for i, model in enumerate(model_list):
    model.fit(x_train,y_train)

    acc = model.score(x_test,y_test)
    print(type(model).__name__,"`s ACC: ",acc,sep='')
    print(type(model).__name__, ":",model.feature_importances_, "\n")
    plot_FI(model)

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