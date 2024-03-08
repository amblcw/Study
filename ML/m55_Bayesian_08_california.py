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
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

import warnings
warnings.filterwarnings(action='ignore')
#data

x, y = fetch_california_housing(return_X_y=True)

# print(np.unique(y,return_counts=True)) # 회귀 
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

y = LabelEncoder().fit_transform(y)
print(x.shape,y.shape)
print(np.unique(y,return_counts=True))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, train_size=0.8,
    # stratify=y
)



sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)

# model 
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
params = {
    'learning_rate':(0.001,1),
    'max_depth':(3,10),
    'num_leaves':(24,40),
    'min_child_samples':(10,200),
    'min_child_weight':(1,50),
    'subsample':(0.5,1),
    'colsample_bytree':(0.5,1),
    'max_bin':(9,500),
    'reg_lambda':(0.001,10),
    'reg_alpha':(0.01,50),
}

def xgb_function(learning_rate,max_depth,num_leaves,min_child_samples,min_child_weight,subsample,colsample_bytree,max_bin,reg_lambda,reg_alpha):
    params={
        'n_estimators':100,
        'learning_rate':learning_rate,
        'max_depth':int(round(max_depth)),
        'num_leaves':int(round(num_leaves)),
        'min_child_samples':int(round(min_child_samples)),
        'min_child_weight':int(round(min_child_weight)),
        'subsample':max(min(subsample,1),0),
        'colsample_bytree':colsample_bytree,
        'max_bin':max(int(round(max_bin)),10),
        'reg_lambda':reg_lambda,
        'reg_alpha':reg_alpha,
        'n_jobs':-1,
    }
    
    model = XGBRegressor(**params)
    model.fit(x_train,y_train,
              eval_set=[(x_train,y_train),(x_test,y_test)],
            #   eval_metric='mlogloss',
              verbose=0,
              early_stopping_rounds=50,
              )
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test,y_pred)
    return r2

from bayes_opt import BayesianOptimization
bay = BayesianOptimization(f=xgb_function,
                           pbounds=params,
                           random_state=47,
                           )

import time
N_ITER = 50
st = time.time()
bay.maximize(init_points=5,n_iter=N_ITER)
et = time.time()

print(bay.max)
print(N_ITER,'번 걸린시간: ',round(et-st,2))

# Score:  0.5915697335360739
# ACC:  0.5915697335360739

# VotingRegressor
# ACC:  0.7746775554623502

# {'target': 0.8398964650126803, 'params': {'colsample_bytree': 0.9347732537990507, 'learning_rate': 0.23042045397016467, 'max_bin': 96.33555897864606, 'max_depth': 5.593272198078207, 'min_child_samples': 31.211563139360827, 'min_child_weight': 20.791638717010905, 'num_leaves': 38.32531211994699, 'reg_alpha': 42.98429885642512, 'reg_lambda': 6.746843144674079, 'subsample': 0.9780433890072424}}
# 50 번 걸린시간:  9.81