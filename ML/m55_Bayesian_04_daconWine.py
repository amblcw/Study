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

# print(x.shape,y.shape)  #(5497, 12) (5497,)
print(np.unique(y,return_counts=True))
# print(y.shape)          #(5497, 7)

x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0
# print(x)
test_csv.loc[test_csv['type'] == 'red', 'type'] = 1 
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0

from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

y = LabelEncoder().fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, train_size=0.8,
    stratify=y
)

print(x_train.shape,y_train.shape)
print(np.unique(y_train,return_counts=True))


sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)

# model 
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
    
    model = XGBClassifier(**params)
    model.fit(x_train,y_train,
              eval_set=[(x_train,y_train),(x_test,y_test)],
            #   eval_metric='mlogloss',
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

# Score:  0.5454545454545454
# ACC:  0.5454545454545454

# VotingClassifier hard
# ACC:  0.6763636363636364

# VotingClassifier soft
# ACC:  0.6754545454545454

# {'target': 0.6645454545454546, 'params': {'colsample_bytree': 0.837589392553407, 'learning_rate': 0.4574236655331919, 'max_bin': 185.37295450018817, 'max_depth': 9.506841040566227, 'min_child_samples': 24.244291078879236, 'min_child_weight': 4.16461371607136, 'num_leaves': 28.615778286902476, 'reg_alpha': 2.7302600375306505, 'reg_lambda': 6.034888341516653, 'subsample': 0.8556415559419639}}
# 100 번 걸린시간:  31.25