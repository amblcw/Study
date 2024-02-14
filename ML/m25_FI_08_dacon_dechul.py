from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import f1_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import time
from sklearn.svm import LinearSVC

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

''' 25퍼 미만 열 삭제 '''
# columns = datasets.feature_names
columns = x.columns
x = pd.DataFrame(x,columns=columns)
print("x.shape",x.shape)
''' 이 밑에 숫자에 얻은 feature_importances 넣고 줄 끝마다 \만 붙여주기'''
fi_str = "1.89481583e-02 1.12573081e-01 3.72493071e-05 8.62489322e-04\
 1.63976671e-02 6.71188692e-03 1.36131405e-03 5.90038758e-03\
 9.48149687e-04 3.94208833e-01 4.42022364e-01 1.15162202e-05\
 1.69034509e-05"
 
''' str에서 숫자로 변환하는 구간 '''
fi_str = fi_str.split()
fi_float = [float(s) for s in fi_str]
print(fi_float)
fi_list = pd.Series(fi_float)

''' 25퍼 미만 인덱스 구하기 '''
low_idx_list = fi_list[fi_list <= fi_list.quantile(0.25)].index
print('low_idx_list',low_idx_list)

''' 25퍼 미만 제거하기 '''
low_col_list = [x.columns[index] for index in low_idx_list]
# 이건 혹여 중복되는 값들이 많아 25퍼이상으로 넘어갈시 25퍼로 자르기
if len(low_col_list) > len(x.columns) * 0.25:   
    low_col_list = low_col_list[:int(len(x.columns)*0.25)]
print('low_col_list',low_col_list)
x.drop(low_col_list,axis=1,inplace=True)
print("after x.shape",x.shape)

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

# r=657
#  LOSS: 1237.2230224609375
# ACC:  0.5243353843688965

# MinMaxScaler
# F1:  0.8378815912825547

# StandardScaler
# F1:  0.8308539115878224

# MaxAbsScaler
# F1:  0.8334220011728465

# RobustScaler
# F1:  0.8429713541136693

# ACC list:  [0.4196, 0.3526, 0.5284, 0.4595, 0.8271, 0.7964]
# Best ML:  DecisionTreeClassifier

# ACC:  [0.83394776 0.83560933 0.83150735 0.8307197  0.83274483]
# 평균 ACC: 0.8329

# 최적의 매개변수:  RandomForestClassifier(min_samples_split=3)
# 최적의 파라미터:  {'min_samples_split': 3}
# best score:  0.7986337531735541
# model score:  0.8123572170301142
# ACC:  0.8123572170301142
# y_pred_best`s ACC: 0.8123572170301142
# time: 147.48sec

# 최적의 매개변수:  RandomForestClassifier(min_samples_split=3, n_jobs=-1)
# 최적의 파라미터:  {'min_samples_split': 3, 'n_jobs': -1}
# best score:  0.8008146289202148
# model score:  0.8066458982346832
# ACC:  0.8065420560747664
# y_pred_best`s ACC: 0.8065420560747664
# time: 181.68sec

# Random
# acc :  0.7813489796978036
# time:  30.677210092544556 sec

# acc :  0.8014434809699361
# time:  1.346724033355713 sec

# DecisionTreeClassifier`s ACC: 0.8320265849732592
# DecisionTreeClassifier : [6.01302079e-02 3.38371608e-02 1.37347326e-02 6.00958829e-03
#  3.67182523e-02 3.27144238e-02 2.38436504e-02 8.13204714e-03
#  5.95981593e-03 4.23887361e-01 3.54312083e-01 2.87473965e-04
#  4.33202630e-04]

# RandomForestClassifier`s ACC: 0.8055454592657978
# RandomForestClassifier : [0.09978479 0.02769677 0.0452933  0.0168483  0.08239856 0.09002902
#  0.07129184 0.02561724 0.01648257 0.26350757 0.25967466 0.00048127
#  0.00089411]

# GradientBoostingClassifier`s ACC: 0.7463523547432369
# GradientBoostingClassifier : [1.89481583e-02 1.12573081e-01 3.72493071e-05 8.62489322e-04
#  1.63976671e-02 6.71188692e-03 1.36131405e-03 5.90038758e-03
#  9.48149687e-04 3.94208833e-01 4.42022364e-01 1.15162202e-05
#  1.69034509e-05]

# XGBClassifier`s ACC: 0.8533153330910224
# XGBClassifier : [0.04650098 0.41125202 0.01180101 0.01636909 0.0325508  0.01734266
#  0.01371949 0.02511661 0.01913969 0.18784563 0.19910412 0.01163159
#  0.00762636]

# after
# DecisionTreeClassifier`s ACC: 0.836024715717327
# DecisionTreeClassifier : [6.33324092e-02 3.38371608e-02 3.97277801e-02 3.69644380e-02
#  2.80491764e-02 9.04587947e-03 6.31328061e-03 4.26060169e-01
#  3.56356924e-01 3.12782689e-04]

# RandomForestClassifier`s ACC: 0.8458902331377538
# RandomForestClassifier : [0.1036061  0.02793278 0.07548263 0.08144864 0.06341335 0.02215726
#  0.01473566 0.318722   0.29174761 0.00075398]

# GradientBoostingClassifier`s ACC: 0.7499350952801288
# GradientBoostingClassifier : [1.87568424e-02 1.09154681e-01 1.68592949e-02 6.85544837e-03
#  1.56947559e-03 6.16161886e-03 9.82762216e-04 3.99324376e-01
#  4.40318643e-01 1.68572856e-05]

# XGBClassifier`s ACC: 0.8566903785243263
# XGBClassifier : [0.04684655 0.43226427 0.03187074 0.01740447 0.01369627 0.02478606
#  0.0180938  0.19347166 0.21088879 0.01067743]