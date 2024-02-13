from sklearn.datasets import load_iris

x, y = load_iris(return_X_y=True)

from m15_HalvingSearch_00 import m15_classifier
m15_classifier(x,y, min_resources=15)

# 최적의 매개변수:  RandomForestClassifier(max_depth=6, min_samples_leaf=3, n_estimators=200)
# 최적의 파라미터:  {'max_depth': 6, 'min_samples_leaf': 3, 'n_estimators': 200}
# best score:  0.9555555555555555
# model score:  1.0
# ACC:  1.0
# y_pred_best`s ACC: 1.0
# time: 2.85sec

# acc :  0.9666666666666667
# time:  1.7793960571289062 sec

# n_iterations: 3
# n_required_iterations: 4
# n_possible_iterations: 3
# min_resources_: 10
# max_resources_: 120
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 52
# n_resources: 10
# Fitting 5 folds for each of 52 candidates, totalling 260 fits
# ----------
# iter: 1
# n_candidates: 18
# n_resources: 30
# Fitting 5 folds for each of 18 candidates, totalling 90 fits
# ----------
# iter: 2
# n_candidates: 6
# n_resources: 90
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# R2 :  0.9491690204159742
# time:  22.16383123397827 sec