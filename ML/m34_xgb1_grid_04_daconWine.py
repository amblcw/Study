from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.datasets import fetch_covtype
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
import numpy as np

path = "C:\\_data\\DACON\\와인품질분류\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submit_csv = pd.read_csv(path+"sample_submission.csv")

# print(train_csv.isna().sum(),test_csv.isna().sum()) 결측치 존재안함

# print(train_csv,test_csv,sep='\n') #[5497 rows x 13 columns], [1000 rows x 12 columns]
x = train_csv.drop(['quality'],axis=1)
y = train_csv['quality']

print(np.unique(y,return_counts=True))

# print(x.shape,y.shape)  #(5497, 12) (5497,)
print(np.unique(y,return_counts=True))
# print(y.shape)          #(5497, 7)

x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0
# print(x)
test_csv.loc[test_csv['type'] == 'red', 'type'] = 1 
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0

import matplotlib.pyplot as plt
plt.yscale('symlog')
plt.boxplot(x)
plt.show()

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
    stratify=y
)

sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)

N_SPLITS = 5    
kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=333)
# kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=333)

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
    'n_estimators'      : [100,200,300,400,500,1000],
    'learning_rate'     : [0.01,0.03,0.05,0.07,0.1,0.3,0.5,1,3],    # eta
    'max_depth'         : [None,2,3,4,5,6,7,8,9,10],
    'gamma'             : [0,1,2,3,4,5,7,10,100],
    'min_child_weight'  : [0,0.01,0.001,0.1,0.5,1,5,10,100],
    # 'early_stoppint_rounds' : [50],
    # 'tree_method'       : ['hist'],
    # 'device'            : ['cuda'],
    # 'reg_alpth'         : [0,0.1,0.01,0.001,1,2,10],
    # 'reg_lamda'         : [0,0.1,0.01,0.001,1,2,10],
}

# model
xgb = XGBClassifier(random_state=333)
model = RandomizedSearchCV(xgb, parameters, refit=True, cv=kfold, n_iter=50, n_jobs=22)

# fit
model.fit(x_train,y_train)

# evaluate
print("best param : ",model.best_params_)
y_predict = model.best_estimator_.predict(x_test)

acc = accuracy_score(y_test, y_predict)
print("ACC score  : ",acc)

# best param :  {'n_estimators': 400, 'min_child_weight': 0.01, 'max_depth': 6, 'learning_rate': 0.1, 'gamma': 0, 'early_stoppint_rounds': 50}
# ACC score  :  0.9527777777777777