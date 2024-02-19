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

import matplotlib.pyplot as plt
plt.yscale('symlog')
plt.boxplot(x)
plt.show()

def fit_outlier(data):  
    label_list = []
    for label in data:
        # print("회귀 분류 판단: ",len(np.unique(data[label])))
        if len(np.unique(data[label])) > 10:
            label_list.append(label)
    
    data = pd.DataFrame(data)
    for label in label_list:
        series = data[label].copy()
        q1 = series.quantile(0.25)      
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + iqr
        lower_bound = q1 - iqr
        
        series[series > upper_bound] = np.nan
        series[series < lower_bound] = np.nan
        print(f"{label:<20}의 이상치 개수: ",series.isna().sum())
        series = series.interpolate()
        data[label] = series
        
    data = data.fillna(data.ffill())
    data = data.fillna(data.bfill())
    return data

x = fit_outlier(x)
# print(x.isna().sum())

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, train_size=0.8,
    stratify=y
)

sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)

N_SPLITS = 5    
kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=333)
# kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=333)

'''
'n_estimators'      : [100,200,300,400,500,1000] default 100            | 1~inf
'learning_rate'     : [0.01,0.03,0.05,0.07,0.1,0.3,0.5,1] default 0.3   | 0~1
'max_depth'         : [None,2,3,4,5,6,7,8,9,10] default 6               | 0~inf
'gamma'             : [0,1,2,3,4,5,7,10,100] default 0                  | 0~inf
'min_child_weight'  : [0,0.01,0.001,0.1,0.5,1,5,10,100] default 1       | 0~inf
'subsample'         : [0,0.1,0.2,0.3,0.5,0.7,1] default 1               | 0~1
'colsample_bytree'  : [0,0.1,0.2,0.3,0.5,0.7,1] default 1               | 0~1
'colsample_bylevel' : [0,0.1,0.2,0.3,0.5,0.7,1] default 1               | 0~1
'colsample_bynode'  : [0,0.1,0.2,0.3,0.5,0.7,1] default 1               | 0~1
'reg_alpth'         : [0,0.1,0.01,0.001,1,2,10] default 1               | 0~inf | L1 절대값 가중치 규제 alpha
'reg_lamda'         : [0,0.1,0.01,0.001,1,2,10] defalut 1               | 0~inf | L2 절대값 가중치 규제 lamda
'''
parameters = {
    'n_estimators'      : [100,200,300,400,500,1000],
    'learning_rate'     : [0.01,0.03,0.05,0.07,0.1,0.3,0.5,1,3],    # eta
    'max_depth'         : [None,2,3,4,5,6,7,8,9,10],
    'gamma'             : [0,1,2,3,4,5,7,10,100],
    'min_child_weight'  : [0,0.01,0.001,0.1,0.5,1,5,10,100],
    # 'early_stoppint_rounds' : [50],
    # 'tree_method'       : ['hist'],
    # 'device'            : ['cuda'],
    # 'reg_alpth'         : [0,0.1,0.01,0.001,1,2,10],
    # 'reg_lamda'         : [0,0.1,0.01,0.001,1,2,10],
}

# model
xgb = XGBClassifier(random_state=333)
model = RandomizedSearchCV(xgb, parameters, refit=True, cv=kfold, n_iter=50, n_jobs=22)

# fit
model.fit(x_train,y_train)

# evaluate
print("best param : ",model.best_params_)
y_predict = model.best_estimator_.predict(x_test)

acc = accuracy_score(y_test, y_predict)
print("ACC score  : ",acc)

# best param :  {'n_estimators': 1000, 'min_child_weight': 0, 'max_depth': 7, 'learning_rate': 0.3, 'gamma': 0}
# ACC score  :  0.8060646970247677