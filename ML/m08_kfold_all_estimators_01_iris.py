import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier

# data
x, y = load_iris(return_X_y=True)

from m08_addon import m08_classifier
m08_classifier(x,y)

# evaluate
# ACC:  [1.         0.96666667 0.93333333 1.         0.9       ]
# 평균 ACC: 0.96