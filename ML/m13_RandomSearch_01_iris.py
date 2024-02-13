from sklearn.datasets import load_iris

x, y = load_iris(return_X_y=True)

from m13_RandomSearch_00 import m13_classifier
m13_classifier(x,y)

# 최적의 매개변수:  RandomForestClassifier(max_depth=6, min_samples_leaf=3, n_estimators=200)
# 최적의 파라미터:  {'max_depth': 6, 'min_samples_leaf': 3, 'n_estimators': 200}
# best score:  0.9555555555555555
# model score:  1.0
# ACC:  1.0
# y_pred_best`s ACC: 1.0
# time: 2.85sec

# acc :  0.9666666666666667
# time:  1.7793960571289062 sec