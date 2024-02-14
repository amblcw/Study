from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import sklearn.preprocessing

import warnings
warnings.filterwarnings(action='ignore')

#data
# x, y = load_iris(return_X_y=True)
dataset = load_iris()
x = dataset.data
y = dataset.target 
columns = dataset.feature_names
# print(f"{x.shape=}, {y.shape=}")        #x.shape=(150, 4), y.shape=(150,)
# print(np.unique(y, return_counts=True)) #array([0, 1, 2]), array([50, 50, 50])

# x = np.delete(x, 0, axis=1)

x = pd.DataFrame(x, columns=columns)
print(x.head)
x = x.drop(['sepal length (cm)'],axis=1)
print(x.head)

x_train , x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=333,stratify=y)

#model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

param = {'random_state':123}
model_list = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

for model in model_list:
    model.fit(x_train,y_train)

    acc = model.score(x_test,y_test)
    print(type(model).__name__,"`s ACC: ",acc,sep='')
    print(type(model).__name__, ":",model.feature_importances_, "\n")

# DecisionTreeClassifier(random_state=123)`s ACC: 0.9333333333333333
# DecisionTreeClassifier(random_state=123) : [0.         0.00625    0.55101744 0.44273256]

# RandomForestClassifier(random_state=123)`s ACC: 0.9333333333333333
# RandomForestClassifier(random_state=123) : [0.08114023 0.01963669 0.45054473 0.44867835]

# GradientBoostingClassifier(random_state=123)`s ACC: 0.9333333333333333
# GradientBoostingClassifier(random_state=123) : [0.00359863 0.00400242 0.36180098 0.63059797]

# 삭제후
# DecisionTreeClassifier`s ACC: 0.9
# DecisionTreeClassifier : [0.00625    0.56351744 0.43023256]

# RandomForestClassifier`s ACC: 0.9333333333333333
# RandomForestClassifier : [0.12428821 0.45546707 0.42024472]

# GradientBoostingClassifier`s ACC: 0.9333333333333333
# GradientBoostingClassifier : [0.00524797 0.27834044 0.71641159]

# XGBClassifier`s ACC: 0.9
# XGBClassifier : [0.0090066  0.33300284 0.6579906 ]