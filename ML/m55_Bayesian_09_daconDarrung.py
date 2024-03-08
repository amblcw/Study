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

# Score:  0.5878802282973639
# R2:  0.5878802282973639

# VotingRegressor
# R2:  0.76919554321965

# {'target': 0.815081979067602, 'params': {'colsample_bytree': 0.700905733548923, 'learning_rate': 0.27851638379965743, 'max_bin': 180.55182732552106, 'max_depth': 5.51384235092177, 'min_child_samples': 18.5892342393525, 'min_child_weight': 3.171729078256951, 'num_leaves': 31.268157568260662, 'reg_alpha': 3.786381057559581, 'reg_lambda': 2.774119644233815, 'subsample': 0.9148507234206205}}
# 50 번 걸린시간:  4.92