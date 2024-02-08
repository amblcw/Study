from sklearn.datasets import load_iris

x, y = load_iris(return_X_y=True)

from m10_addon import m10
m10(x,y,'c')

# 최적의 매개변수:  RandomForestClassifier(max_depth=6, min_samples_leaf=3, n_estimators=200)
# 최적의 파라미터:  {'max_depth': 6, 'min_samples_leaf': 3, 'n_estimators': 200}
# best score:  0.9555555555555555
# model score:  1.0
# ACC:  1.0
# y_pred_best`s ACC: 1.0
# time: 2.85sec