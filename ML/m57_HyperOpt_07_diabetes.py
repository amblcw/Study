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
from sklearn.datasets import load_diabetes

import warnings
warnings.filterwarnings(action='ignore')
#data

x, y = load_diabetes(return_X_y=True)

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
from hyperopt import *
search_space = {
    'n_estimators':100,
    'learning_rate':hp.uniform('learning_rate',0.001,0.1),
    'max_depth':hp.quniform('max_depth',3,10,1),
    'num_leaves':hp.quniform('num_leaves',24,40,1),
    'min_child_samples':hp.quniform('min_child_samples',10,200,1),
    'min_child_weight':hp.quniform('min_child_weight',10,200,1),
    'subsample':hp.uniform('subsample',0.5,1),
    'colsample_bytree':hp.uniform('colsample_bytree',0.5,1),
    'max_bin':hp.quniform('max_bin',9,500,1),
    # 'reg_lambda':hp.uniform('reg_lambda',-0.001,10),
    # 'reg_alpha':hp.uniform('reg_lambda',-0.001,10),
    'n_jobs':-1,
}

def xgb_function(serch_space=search_space):
    params={
        'n_estimators':100,
        'learning_rate':search_space['learning_rate'],
        'max_depth':search_space['max_depth'],
        'num_leaves':search_space['num_leaves'],
        'min_child_samples':search_space['min_child_samples'],
        'min_child_weight':search_space['min_child_weight'],
        'subsample':search_space['subsample'],
        'colsample_bytree':search_space['colsample_bytree'],
        'max_bin':search_space['max_bin'],
        # 'reg_lambda':search_space['reg_lambda'],
        # 'reg_alpha':search_space['reg_alpha'],
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
    loss = model.score(x_test,y_test)
    return loss

trial_val = Trials()

best = fmin(
    fn= xgb_function, # 목적함수
    space= search_space,    # 탐색범위
    algo= tpe.suggest,      # 알고리즘, default
    max_evals= 50,          # 탐색횟수
    trials= trial_val,      
    rstate= np.random.default_rng(seed=10)  # random state
)

print(best)

# Score:  0.4602452781304722
# ACC:  0.4602452781304722

# VotingRegressor
# ACC:  0.37375396432291663

# {'target': 0.4779106377619179, 'params': {'colsample_bytree': 0.720419276860619, 'learning_rate': 0.4129337348331773, 'max_bin': 14.024096409841057, 'max_depth': 4.924746015604292, 'min_child_samples': 144.85575444238552, 'min_child_weight': 43.29750560877663, 'num_leaves': 34.18832581640202, 'reg_alpha': 25.7957390786402, 'reg_lambda': 7.994484531861016, 'subsample': 0.8484372910772067}}
# 50 번 걸린시간:  4.53

# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:01<00:00, 28.38trial/s, best loss: 0.31789174804295195]
# {'colsample_bytree': 0.8286102646336342, 'learning_rate': 0.029773609448724817, 'max_bin': 216.0, 'max_depth': 7.0, 'min_child_samples': 73.0, 'min_child_weight': 175.0, 'num_leaves': 30.0, 'subsample': 0.9299459818100199}