from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
a = 10
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

x = np.array([
    [1,2,3,4,5],
    [2,3,4,5,6],
    [3,4,5,6,7],
    ]).T

y = np.array([
    [10,20,30,40,50],
    [20,30,40,50,60],
]).T

model = RandomForestRegressor()
model.fit(x,y)
r2 = model.score(x,y)
pred = model.predict(x)
print("r2: ",r2)
print(pred)