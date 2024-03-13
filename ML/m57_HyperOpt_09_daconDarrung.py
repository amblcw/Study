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
path = "C:\\_data\\DACON\\ddarung\\"
train_csv = pd.read_csv(path+"train.csv",index_col=['id'])  
test_csv = pd.read_csv(path+"test.csv",index_col=0)         
submission_csv = pd.read_csv(path+"submission.csv")

train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'],axis=1) #count 를 드랍, axis=0은 행, axis=1은 열
y = train_csv['count']

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

# Score:  0.5878802282973639
# R2:  0.5878802282973639

# VotingRegressor
# R2:  0.76919554321965

# {'target': 0.815081979067602, 'params': {'colsample_bytree': 0.700905733548923, 'learning_rate': 0.27851638379965743, 'max_bin': 180.55182732552106, 'max_depth': 5.51384235092177, 'min_child_samples': 18.5892342393525, 'min_child_weight': 3.171729078256951, 'num_leaves': 31.268157568260662, 'reg_alpha': 3.786381057559581, 'reg_lambda': 2.774119644233815, 'subsample': 0.9148507234206205}}
# 50 번 걸린시간:  4.92

# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:03<00:00, 15.95trial/s, best loss: 0.8069056066183327]
# {'colsample_bytree': 0.8286102646336342, 'learning_rate': 0.029773609448724817, 'max_bin': 216.0, 'max_depth': 7.0, 'min_child_samples': 73.0, 'min_child_weight': 175.0, 'num_leaves': 30.0, 'subsample': 0.9299459818100199}