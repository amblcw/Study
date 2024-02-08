import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC 

# data
x, y = load_iris(return_X_y=True)



from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold

N_SPLIT = 5
kfold = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=123)

# model
model = SVC()

# fit
scores = cross_val_score(model, x, y, cv=kfold)
print("ACC: ",scores)
print(f"평균 ACC: {round(np.mean(scores),4)}")

# evaluate
# ACC:  [1.         0.96666667 0.93333333 1.         0.9       ]
# 평균 ACC: 0.96