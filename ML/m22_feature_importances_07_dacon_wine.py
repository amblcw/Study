from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import sklearn.preprocessing
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings(action='ignore')

path = "C:\\_data\\DACON\\와인품질분류\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submit_csv = pd.read_csv(path+"sample_submission.csv")

# print(train_csv.isna().sum(),test_csv.isna().sum()) 결측치 존재안함

# print(train_csv,test_csv,sep='\n') #[5497 rows x 13 columns], [1000 rows x 12 columns]
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
train_csv['type'] = label_encoder.fit_transform(train_csv['type'])
train_csv['quality'] = label_encoder.fit_transform(train_csv['quality'])

print(train_csv['quality'])
print(train_csv.head)
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



# print(test_csv)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=123)
param = {'random_state':123}
model_list = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

for model in model_list:
    model.fit(x_train,y_train)

    acc = model.score(x_test,y_test)
    print(type(model).__name__,"`s ACC: ",acc,sep='')
    print(type(model).__name__, ":",model.feature_importances_, "\n")

# StandardScaler
# LOSS: 1.0394665002822876
#  ACC:  0.5745454545454546(0.5745454430580139 by loss[1])

# ACC:  [0.44454545 0.44181818 0.44040036 0.43949045 0.43949045]
# 평균 ACC: 0.4411

# 최적의 매개변수:  RandomForestClassifier(min_samples_split=5, n_jobs=-1)
# 최적의 파라미터:  {'min_samples_split': 5, 'n_jobs': -1}
# best score:  0.6634329135643595
# model score:  0.6981818181818182
# ACC:  0.6981818181818182
# y_pred_best`s ACC: 0.6981818181818182
# time: 9.19sec

# Random
# acc :  0.6181818181818182
# time:  3.7730820178985596 sec

# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 4
# min_resources_: 162
# max_resources_: 4397
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 52
# n_resources: 162
# Fitting 5 folds for each of 52 candidates, totalling 260 fits
# ----------
# iter: 1
# n_candidates: 18
# n_resources: 486
# Fitting 5 folds for each of 18 candidates, totalling 90 fits
# ----------
# iter: 2
# n_candidates: 6
# n_resources: 1458
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# ----------
# iter: 3
# n_candidates: 2
# n_resources: 4374
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# acc :  0.65
# time:  35.686814069747925 sec

# acc :  0.65
# time:  0.14526844024658203 sec

# DecisionTreeClassifier`s ACC: 0.5754545454545454
# DecisionTreeClassifier : [0.06746041 0.11007581 0.07139835 0.09299866 0.0788758  0.09011881
#  0.08798086 0.09026043 0.08147442 0.09051516 0.13795934 0.00088197]

# RandomForestClassifier`s ACC: 0.6627272727272727
# RandomForestClassifier : [0.07313185 0.09886063 0.07958795 0.08317828 0.08574706 0.08688293
#  0.09004553 0.10461417 0.08383385 0.08769391 0.12226881 0.00415504]

# GradientBoostingClassifier`s ACC: 0.5418181818181819
# GradientBoostingClassifier : [0.01599918 0.08791371 0.04949751 0.05671506 0.3370065  0.03474825
#  0.03315705 0.06492459 0.04794481 0.04624764 0.22141076 0.00443494]

# XGBClassifier`s ACC: 0.6327272727272727
# XGBClassifier : [0.05545606 0.08902279 0.05532062 0.06314326 0.05454681 0.06951984
#  0.06095583 0.05880075 0.05955473 0.06543217 0.17134504 0.19690208]