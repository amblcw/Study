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

import warnings
warnings.filterwarnings(action='ignore')

path = "C:\\_data\\DACON\\loan\\"

train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submission_csv = pd.read_csv(path+"sample_submission.csv")

# print(train_csv.shape, test_csv.shape) #(96294, 14) (64197, 13)
# print(train_csv.columns, test_csv.columns,sep='\n',end="\n======================\n")
# Index(['대출금액', '대출기간', '근로기간', '주택소유상태', '연간소득', '부채_대비_소득_비율', '총계좌수', '대출목적',
#        '최근_2년간_연체_횟수', '총상환원금', '총상환이자', '총연체금액', '연체계좌수', '대출등급'],
#       dtype='object')
# Index(['대출금액', '대출기간', '근로기간', '주택소유상태', '연간소득', '부채_대비_소득_비율', '총계좌수', '대출목적',
#        '최근_2년간_연체_횟수', '총상환원금', '총상환이자', '총연체금액', '연체계좌수'],
#       dtype='object')

# print(np.unique(train_csv['주택소유상태'],return_counts=True))
# print(np.unique(test_csv['주택소유상태'],return_counts=True),end="\n======================\n")
# (array(['ANY', 'MORTGAGE', 'OWN', 'RENT'], dtype=object), array([    1, 47934, 10654, 37705], dtype=int64))
# (array(['MORTGAGE', 'OWN', 'RENT'], dtype=object), array([31739,  7177, 25281], dtype=int64))

# print(np.unique(train_csv['대출목적'],return_counts=True))
# print(np.unique(test_csv['대출목적'],return_counts=True),end="\n======================\n")
# (array(['기타', '부채 통합', '소규모 사업', '신용 카드', '의료', '이사', '자동차', '재생 에너지',
#        '주요 구매', '주택', '주택 개선', '휴가'], dtype=object), array([ 4725, 55150,   787, 24500,  1039,   506,   797,    60,  1803,
#          301,  6160,   466], dtype=int64))
# (array(['결혼', '기타', '부채 통합', '소규모 사업', '신용 카드', '의료', '이사', '자동차',
#        '재생 에너지', '주요 구매', '주택', '주택 개선', '휴가'], dtype=object), array([    1,  3032, 37054,   541, 16204,   696,   362,   536,    29,
#         1244,   185,  4019,   294], dtype=int64))

# print(np.unique(train_csv['대출등급'],return_counts=True),end="\n======================\n")
# (array(['A', 'B', 'C', 'D', 'E', 'F', 'G'], dtype=object), array([16772, 28817, 27623, 13354,  7354,  1954,   420], dtype=int64))

train_csv = train_csv[train_csv['주택소유상태'] != 'ANY'] #ANY은딱 한개 존재하기에 그냥 제거
# test_csv = test_csv[test_csv['대출목적'] != '결혼']
test_csv.loc[test_csv['대출목적'] == '결혼' ,'대출목적'] = '기타' #결혼은 제거하면 개수가 안맞기에 기타로 대체

# x.loc[x['type'] == 'red', 'type'] = 1
# print(np.unique(train_csv['주택소유상태'],return_counts=True))
# print(np.unique(test_csv['주택소유상태'],return_counts=True),end="\n======================\n")
# print(np.unique(train_csv['대출목적'],return_counts=True))
# print(np.unique(test_csv['대출목적'],return_counts=True),end="\n======================\n")

#대출기간 처리
train_csv['대출기간'] = train_csv['대출기간'].replace({' 36 months' : 36 , ' 60 months' : 60 }).astype(int)
test_csv['대출기간'] = test_csv['대출기간'].replace({' 36 months' : 36 , ' 60 months' : 60 }).astype(int)
# train_loan_time = train_csv['대출기간']
# train_loan_time = train_loan_time.str.split()
# for i in range(len(train_loan_time)):
#     train_loan_time.iloc[i] = int(train_loan_time.iloc[i][0]) #앞쪽 숫자만 따서 int로 변경
  
# train_csv['대출기간'] = train_loan_time 
    
# test_loan_time = test_csv['대출기간']
# test_loan_time = test_loan_time.str.split()
# for i in range(len(test_loan_time)):
#     test_loan_time.iloc[i] = int(test_loan_time.iloc[i][0]) #앞쪽 숫자만 따서 int로 변경    

# test_csv['대출기간'] = test_loan_time

#근로기간 처리
train_working_time = train_csv['근로기간']
test_working_time = test_csv['근로기간']

for i in range(len(train_working_time)):
    data = train_working_time.iloc[i]
    if data == 'Unknown':
        train_working_time.iloc[i] = np.NaN
    elif data == '10+ years' or data == '10+years':
        train_working_time.iloc[i] = int(30)
    elif data == '< 1 year' or data == '<1 year':
        train_working_time.iloc[i] = int(0)
    else:
        train_working_time.iloc[i] = int(data.split()[0])
    
train_working_time = train_working_time.fillna(train_working_time.mean())

for i in range(len(test_working_time)):
    data = test_working_time.iloc[i]
    if data == 'Unknown':
        test_working_time.iloc[i] = np.NaN
    elif data == '10+ years' or data == '10+years':
        test_working_time.iloc[i] = int(30)
    elif data == '< 1 year' or data == '<1 year':
        test_working_time.iloc[i] = int(0)
    else:
        test_working_time.iloc[i] = int(data.split()[0])
    
test_working_time = test_working_time.fillna(test_working_time.mean())

train_csv['근로기간'] = train_working_time
test_csv['근로기간'] = test_working_time 

#주택소유상태 처리

trian_have_house = train_csv['주택소유상태']
label_encoder = LabelEncoder()
trian_have_house = label_encoder.fit_transform(trian_have_house)
train_csv['주택소유상태'] = trian_have_house

test_have_house = test_csv['주택소유상태']
label_encoder = LabelEncoder()
test_have_house = label_encoder.fit_transform(test_have_house)
test_csv['주택소유상태'] = test_have_house

#대출목적 처리
trian_loan_purpose = train_csv['대출목적']
label_encoder = LabelEncoder()
trian_loan_purpose = label_encoder.fit_transform(trian_loan_purpose)
train_csv['대출목적'] = trian_loan_purpose

test_loan_purpose = test_csv['대출목적']
label_encoder = LabelEncoder()
test_loan_purpose = label_encoder.fit_transform(test_loan_purpose)
test_csv['대출목적'] = test_loan_purpose

#대출등급 처리
train_loan_grade = train_csv['대출등급']
label_encoder = LabelEncoder()
train_loan_grade = label_encoder.fit_transform(train_loan_grade)
train_csv['대출등급'] = train_loan_grade

# print(train_csv.isna().sum(),test_csv.isna().sum(), sep='\n') #결측치 제거 완료 확인함

# for label in train_csv:                                       #모든 데이터가  또는 실수로 변경됨을 확인함
#     for data in train_csv[label]:
#         if type(data) != type(1) and type(data) != type(1.1):
#             print("not int, not float : ",data)


# for label in test_csv:
#     print(label)
#     print(f"train[{label}]: ",np.unique(train_csv[label],return_counts=True))
#     print(f"test[{label}]",np.unique(test_csv[label],return_counts=True))
x = train_csv.drop(['대출등급'],axis=1)
y = train_csv['대출등급']

print(f"{test_csv.shape=}")
print(np.unique(y,return_counts=True)) #(array([0, 1, 2, 3, 4, 5, 6]), array([16772, 28817, 27622, 13354,  7354,  1954,   420], dtype=int64))

y = y.to_frame(['대출등급'])
# y = y.reshape(-1,1)

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
    
    model = XGBClassifier(**params)
    model.fit(x_train,y_train,
              eval_set=[(x_train,y_train),(x_test,y_test)],
            #   eval_metric='mlogloss',
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

# Score:  0.42619035256243837
# ACC:  0.42619035256243837

# VotingClassifier hard
# ACC:  0.8032088893504336

# VotingClassifier soft
# ACC:  0.8409055506516434

# {'target': 0.8478633366218391, 'params': {'colsample_bytree': 0.920250934126465, 'learning_rate': 0.9114818668999987, 'max_bin': 432.7141104671454, 'max_depth': 5.580022546698496, 'min_child_samples': 96.38686341221957, 'min_child_weight': 14.404526982989198, 'num_leaves': 25.45835596301343, 'reg_alpha': 2.1595973008298133, 'reg_lambda': 2.3896371581898785, 'subsample': 0.9111166887537178}}
# 50 번 걸린시간:  62.28

# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:20<00:00,  1.04s/trial, best loss: 0.8517056960382159]
# {'colsample_bytree': 0.8286102646336342, 'learning_rate': 0.029773609448724817, 'max_bin': 216.0, 'max_depth': 7.0, 'min_child_samples': 73.0, 'min_child_weight': 175.0, 'num_leaves': 30.0, 'subsample': 0.9299459818100199}