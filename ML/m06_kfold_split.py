import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC 

# data
# x, y = load_iris(return_X_y=True)
dataset = load_iris()

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

print(df)
print(df.columns)



from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold

N_SPLIT = 3
kfold = KFold(n_splits=N_SPLIT, shuffle=True, random_state=123)

for train_index, val_index in kfold.split(df):
    print("====================")
    print(train_index,val_index)
    print("훈련데이터 갯수: ",len(train_index), "\n검증데이터 개수: ", len(val_index))

# model
model = SVC()

# fit
# scores = cross_val_score(model, x, y, cv=kfold)
# print("ACC: ",scores)
# print(f"평균 ACC: {round(np.mean(scores),4)}")

# evaluate
# ACC:  [1.         0.96666667 0.93333333 1.         0.9       ]
# 평균 ACC: 0.96