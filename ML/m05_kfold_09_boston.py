from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import load_boston    # pip install scikit-learn==1.1.3
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.svm import LinearSVR

import warnings
warnings.filterwarnings('ignore')

datasets = load_boston()
x = datasets.data
y = datasets.target

r2 = 0

# while r2 < 0.8:
r = int(np.random.uniform(1,1000))
r = 88
from sklearn.svm import SVR 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold

N_SPLIT = 5
kfold = KFold(n_splits=N_SPLIT, shuffle=True, random_state=123)

# model
model = RandomForestRegressor()

# fit
scores = cross_val_score(model, x, y, cv=kfold)
print("ACC: ",scores)
print(f"평균 ACC: {round(np.mean(scores),4)}")

# StandardScaler
# loss=[4.916057586669922, 1.7392468452453613]
# r2=0.9333945146886241
# RMSE: 2.217218335403849

# ACC:  [0.75567667 0.86588395 0.9117018  0.92818409 0.86901632]
# 평균 ACC: 0.8661