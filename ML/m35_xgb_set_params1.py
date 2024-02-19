from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.datasets import load_diabetes
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
import numpy as np

x, y = load_diabetes(return_X_y=True)

import matplotlib.pyplot as plt
plt.yscale('symlog')
plt.boxplot(x)
# plt.show()

def fit_outlier(data):  
    data = pd.DataFrame(data)
    for label in data:
        series = data[label]
        q1 = series.quantile(0.25)      
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + iqr
        lower_bound = q1 - iqr
        
        series[series > upper_bound] = np.nan
        series[series < lower_bound] = np.nan
        print(series.isna().sum())
        series = series.interpolate()
        data[label] = series
        
    data = data.fillna(data.ffill())
    data = data.fillna(data.bfill())
    return data

x = fit_outlier(x)
# print(x.isna().sum())

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, train_size=0.8,
    # stratify=y
)

# sclaer = MinMaxScaler().fit(x_train)
sclaer = StandardScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)

N_SPLITS = 5    
# kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=333)
kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=333)

'''
'n_estimators'      : [100,200,300,400,500,1000] default 100            | 1~inf
'learning_rate'     : [0.01,0.03,0.05,0.07,0.1,0.3,0.5,1] default 0.3   | 0~1
'max_depth'         : [None,2,3,4,5,6,7,8,9,10] default 6               | 0~inf
'gamma'             : [0,1,2,3,4,5,7,10,100] default 0                  | 0~inf
'min_child_weight'  : [0,0.01,0.001,0.1,0.5,1,5,10,100] default 1       | 0~inf
'subsample'         : [0,0.1,0.2,0.3,0.5,0.7,1] default 1               | 0~1
'colsample_bytree'  : [0,0.1,0.2,0.3,0.5,0.7,1] default 1               | 0~1
'colsample_bylevel' : [0,0.1,0.2,0.3,0.5,0.7,1] default 1               | 0~1
'colsample_bynode'  : [0,0.1,0.2,0.3,0.5,0.7,1] default 1               | 0~1
'reg_alpth'         : [0,0.1,0.01,0.001,1,2,10] default 1               | 0~inf | L1 절대값 가중치 규제 alpha
'reg_lamda'         : [0,0.1,0.01,0.001,1,2,10] defalut 1               | 0~inf | L2 절대값 가중치 규제 lamda
'''
parameters = {
    'n_estimators'      : [100,500,1000],
    'learning_rate'     : [0.01,0.05,0.07,0.1,0.5,1,3],     # eta
    'max_depth'         : [None,3,5,7,9],
    'gamma'             : [0,5,10,100],
    'min_child_weight'  : [0,0.01,0.1,1,10],  
    # 'reg_alpth'         : [0,0.1,0.01,0.001,1,2,10],
    # 'reg_lamda'         : [0,0.1,0.01,0.001,1,2,10],
}

# model
xgb = XGBClassifier(random_state=333)
# model = RandomizedSearchCV(xgb, parameters, refit=True, cv=kfold, n_iter=50, n_jobs=22)
model = XGBRFRegressor()
model.set_params(gamma=0.1)

# fit
model.fit(x_train,y_train)

# evaluate
# print("best param : ",model.best_params_)
# y_predict = model.best_estimator_.predict(x_test)
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("R2 score  : ",r2)

model.set_params(eta=0.001,max_depth=15)

# fit
model.fit(x_train,y_train)

# evaluate
# print("best param : ",model.best_params_)
# y_predict = model.best_estimator_.predict(x_test)
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("R2 score  : ",r2)

model.set_params(n_estimators=400)

# fit
model.fit(x_train,y_train)

# evaluate
# print("best param : ",model.best_params_)
# y_predict = model.best_estimator_.predict(x_test)
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("R2 score  : ",r2)

model.set_params(min_child_weight=100)

# fit
model.fit(x_train,y_train)

# evaluate
# print("best param : ",model.best_params_)
# y_predict = model.best_estimator_.predict(x_test)
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("R2 score  : ",r2)
print("사용된 파라미터: ",model.get_params())

# R2 score  :  0.2612580722724248
# R2 score  :  0.2612580722724248
# R2 score  :  0.28193670536651105