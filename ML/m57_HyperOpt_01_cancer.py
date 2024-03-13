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

from hyperopt import *

# data
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,random_state=47)

sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)

# model
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
    
    model = XGBClassifier(**params)
    model.fit(x_train,y_train,
              eval_set=[(x_train,y_train),(x_test,y_test)],
              eval_metric='logloss',
              verbose=0,
              early_stopping_rounds=50,
              )
    y_pred = model.predict(x_test)
    # acc = accuracy_score(y_test,y_pred)
    loss = model.score(x_test,y_test)
    return loss

trial_val = Trials()

best = fmin(
    fn= xgb_function, # 목적함수
    space= search_space,    # 탐색범위
    algo= tpe.suggest,      # 알고리즘, default
    max_evals= 20,          # 탐색횟수
    trials= trial_val,      
    rstate= np.random.default_rng(seed=10)  # random state
)

print(best)

print(trial_val.results)
print(trial_val.vals)


# print('|   iter   |  target  |    x1    |    x2    |')
# print('---------------------------------------------')
# x1_list = trial_val.vals['x1']
# x2_list = trial_val.vals['x2']
# for idx, data in enumerate(trial_val.results):
#     loss = data['loss']
#     print(f'|{idx:^10}|{loss:^10}|{x1_list[idx]:^10}|{x2_list[idx]:^10}|')

# XGBClassifier()
# Score:  0.9649122807017544
# ACC:  0.9649122807017544

# VotingClassifier hard
# ACC:  0.9649122807017544

# VotingClassifier soft
# ACC:  0.9649122807017544

# {'target': 0.9649122807017544, 'params': {'colsample_bytree': 0.5996432157540442, 'learning_rate': 0.7664901663403912, 'max_bin': 270.9707792429199, 'max_depth': 4.384268219739154, 'min_child_samples': 10.643641019768253, 'min_child_weight': 26.43127375296339, 'num_leaves': 26.12118496637413, 'reg_alpha': 32.86667609501238, 'reg_lambda': 5.60544127095163, 'subsample': 0.5010792244249415}}
# 100 번 걸린시간:  12.03