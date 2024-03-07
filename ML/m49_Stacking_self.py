import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import random as rn
import tensorflow as tf
rn.seed(333)
tf.random.set_seed(333)
np.random.seed(333)

# data
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,random_state=47)

sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)

model = StackingClassifier([
    ('xgb',XGBClassifier()),
    ('RF',RandomForestClassifier()),
    ('LG',LogisticRegression()),
],final_estimator=CatBoostClassifier(verbose=0))

def self_Stacking(models:list[tuple], final_model, x_train, x_test, y_train, y_test):
    pred_list = []
    trained_model_dict = {}
    for name, model in models:
        model.fit(x_train,y_train)
        pred = model.predict(x_train)  
        pred_list.append(pred)
        trained_model_dict[name] = model
        
    voted_pred = np.asarray(pred_list).T
    final_model.fit(voted_pred,y_train)
    
    pred_list = []
    print_dict = {}
    for name, model in trained_model_dict.items():
        pred = model.predict(x_test)   
        result = model.score(x_test,y_test)
        pred_list.append(pred)
        print_dict[f'{name} ACC'] = result
        trained_model_dict[name] = model
    
    voted_pred = np.asarray(pred_list).T
    final_result = final_model.score(voted_pred,y_test)
    
    print(print_dict)
    print("스태킹 결과: ",final_result)
    
self_Stacking([
    ('xgb',XGBClassifier()),
    ('RF',RandomForestClassifier()),
    ('LG',LogisticRegression()),
],CatBoostClassifier(verbose=0),x_train,x_test,y_train,y_test)

model.fit(x_train,y_train)
result = model.score(x_test,y_test)
print("sklearn Stacking의 ACC : ",result)

# {'xgb ACC': 0.9473684210526315, 'RF ACC': 0.9473684210526315, 'LG ACC': 0.9824561403508771}
# 스태킹 결과:  0.9649122807017544
# sklearn Stacking의 ACC :  0.9649122807017544