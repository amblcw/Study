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
    'max_bin':(0,500),
    'reg_lambda':(-0.001,10),
    'reg_alpha':(0.01,50),
}

from bayes_opt import BayesianOptimization
# optimizer = BayesianOptimization(f= y_funstion,
#                                  pbounds=params,
#                                  random_state=47,
#                                  )

# XGBClassifier()
# Score:  0.9649122807017544
# ACC:  0.9649122807017544

# VotingClassifier hard
# ACC:  0.9649122807017544

# VotingClassifier soft
# ACC:  0.9649122807017544

