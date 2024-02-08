import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier

# data
x, y = load_iris(return_X_y=True)



from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=123, stratify=y)

N_SPLIT = 5
kfold = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=123)

# model
model = RandomForestClassifier()

# fit
scores = cross_val_score(model, x_train, y_train, cv=kfold)
# print("ACC: ",scores)
# print(f"평균 ACC: {round(np.mean(scores),4)}")

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
acc = accuracy_score(y_test, y_predict)
print("ACC: ", acc)

# evaluate
# ACC:  [1.         0.96666667 0.93333333 1.         0.9       ]
# 평균 ACC: 0.96