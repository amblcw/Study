import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# data
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,random_state=47)

sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)

# model 
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
    
    model = XGBClassifier(**params)
    model.fit(x_train,y_train,
              eval_set=[(x_train,y_train),(x_test,y_test)],
              eval_metric='logloss',
              verbose=0,
              early_stopping_rounds=50,
              )
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test,y_pred)
    return acc

from bayes_opt import BayesianOptimization
bay = BayesianOptimization(f=xgb_function,
                           pbounds=params,
                           random_state=47,
                           )

import time
N_ITER = 100
st = time.time()
bay.maximize(init_points=5,n_iter=N_ITER)
et = time.time()

print(bay.max)
print(N_ITER,'번 걸린시간: ',round(et-st,2))

# XGBClassifier()
# Score:  0.9649122807017544
# ACC:  0.9649122807017544

# VotingClassifier hard
# ACC:  0.9649122807017544

# VotingClassifier soft
# ACC:  0.9649122807017544

# {'target': 0.9649122807017544, 'params': {'colsample_bytree': 0.5996432157540442, 'learning_rate': 0.7664901663403912, 'max_bin': 270.9707792429199, 'max_depth': 4.384268219739154, 'min_child_samples': 10.643641019768253, 'min_child_weight': 26.43127375296339, 'num_leaves': 26.12118496637413, 'reg_alpha': 32.86667609501238, 'reg_lambda': 5.60544127095163, 'subsample': 0.5010792244249415}}
# 100 번 걸린시간:  12.03