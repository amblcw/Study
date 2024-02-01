from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Input, concatenate
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import f1_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import time

import warnings
warnings.filterwarnings(action='ignore')

path = "C:\\_data\\DACON\\loan\\"

train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submission_csv = pd.read_csv(path+"sample_submission.csv")

trian_have_house = train_csv['대출등급']
label_encoder = LabelEncoder()
trian_have_house = label_encoder.fit_transform(trian_have_house)

"""
"""
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

# ''' 이상치 제거 '''
# print(train_csv.quantile(q=0.25))
# print(train_csv.quantile(q=0.75))
# 대출금액            10200000.00
# 대출기간                  36.00
# 근로기간                   3.00
# 주택소유상태                 0.00
# 연간소득            57600000.00
# 부채_대비_소득_비율           12.65
# 총계좌수                  17.00
# 대출목적                   1.00
# 최근_2년간_연체_횟수           0.00
# 총상환원금             307572.00
# 총상환이자             134616.00
# 총연체금액                  0.00
# 연체계좌수                  0.00
# 대출등급                   1.00
# Name: 0.25, dtype: float64
# 대출금액            2.400000e+07
# 대출기간            6.000000e+01
# 근로기간            3.000000e+01
# 주택소유상태          2.000000e+00
# 연간소득            1.128000e+08
# 부채_대비_소득_비율     2.554000e+01
# 총계좌수            3.200000e+01
# 대출목적            3.000000e+00
# 최근_2년간_연체_횟수    0.000000e+00
# 총상환원금           1.055076e+06
# 총상환이자           5.702280e+05
# 총연체금액           0.000000e+00
# 연체계좌수           0.000000e+00
# 대출등급            2.000000e+00
# Name: 0.75, dtype: float64
q1 = train_csv.quantile(q=0.25)
q3 = train_csv.quantile(q=0.75)
iqr = q3 - q1
lower_limit = q1 - 1.5*iqr
upper_limit = q3 + 1.5*iqr

print(lower_limit)
print(upper_limit)    

print(train_csv.max())
print(train_csv.min())
for label in train_csv:
    if label in ['연간소득','부채_대비_소득_비율','총상환원금','총상환이자']:
        lower = lower_limit[label]
        upper = upper_limit[label]
        
        train_csv.loc[train_csv[label]<lower, label] = lower
        train_csv.loc[train_csv[label]>upper, label] = upper

print(train_csv.max())
print(train_csv.min())

x = train_csv.drop(['대출등급'],axis=1)
y = train_csv['대출등급']
cols = x.columns

print(f"{test_csv.shape=}")
print(np.unique(y,return_counts=True)) #(array([0, 1, 2, 3, 4, 5, 6]), array([16772, 28817, 27622, 13354,  7354,  1954,   420], dtype=int64))

y = y.to_frame(['대출등급'])
# y = y.reshape(-1,1)
# ohe = OneHotEncoder(sparse=False)
# y = ohe.fit_transform(y) 

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler().fit(x_train)    #최솟값을 0 최댓값을 1로 스케일링
scaler = StandardScaler().fit(x)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
# scaler = RobustScaler().fit(x_train)    #

x = scaler.transform(x)
test_csv = scaler.transform(test_csv)

scaler = MinMaxScaler().fit(x)    #최솟값을 0 최댓값을 1로 스케일링
# scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
# scaler = RobustScaler().fit(x_train)    #

x = scaler.transform(x)
test_csv = scaler.transform(test_csv)
x = pd.DataFrame(x,columns=cols)

######### input 분리 ######### 
print(x.columns)
# Index(['대출금액', '대출기간', '근로기간', '주택소유상태', '연간소득', '부채_대비_소득_비율', '총계좌수', '대출목적',
#        '최근_2년간_연체_횟수', '총상환원금', '총상환이자', '총연체금액', '연체계좌수']
x1 = x[['대출금액','연간소득','총상환원금','총상환이자']]                         # 수가 크며 분류형이 아닌 데이터들
x2 = x[['최근_2년간_연체_횟수','연체계좌수','부채_대비_소득_비율','총연체금액']]   # 있으면 안좋은 데이터들
x3 = x[['대출기간','근로기간','주택소유상태','총계좌수','대출목적']]              #나머지 데이터들
data_path = "C:\\Study\\ML\\resource\\m01_smote2_dacon_dechul\\"
# np.save(data_path+"x.npy",arr=x)
# np.save(data_path+"y.npy",arr=y)
# np.save(data_path+"test_csv.npy",arr=test_csv)

# x = np.load(data_path+"x.npy")
# y = np.load(data_path+"y.npy")
# test_csv = np.load(data_path+"test_csv.npy")


f1 = 0
r = int(np.random.uniform(1,1000))
r = 529
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(x1,x2,x3,y,train_size=0.9,random_state=r,stratify=y)
print(np.unique(y_train,return_counts=True))
print(np.unique(y_test,return_counts=True))

print(f"{x1_train.shape=}\n{x1_test.shape=}\n{x2_train.shape=}\n{x2_test.shape=}\n{x3_train.shape=}\n{x3_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")
# x1_train.shape=(86663, 4)
# x1_test.shape=(9630, 4)
# x2_train.shape=(86663, 4)
# x2_test.shape=(9630, 4)
# x3_train.shape=(86663, 5)
# x3_test.shape=(9630, 5)
# y_train.shape=(86663, 1)
# y_test.shape=(9630, 1)

# from imblearn.over_sampling import SMOTE, BorderlineSMOTE
# st_time = time.time()
# smote = BorderlineSMOTE(random_state=r)
# print(type(x_train),type(y_train))
# x_train, y_train = smote.fit_resample(x_train,y_train)
# ed_time = time.time()
# print("time: ",ed_time-st_time)
# print(np.unique(y_train,return_counts=True))

def model1():
    input = Input(shape=(4,))
    d1 = Dense(128, activation='swish')(input)
    d2 = Dense(128, activation='swish')(d1)
    d3 = Dense(128, activation='swish')(d2)
    b1 = BatchNormalization()(d3)
    d4 = Dense(64, activation='swish')(b1)
    d5 = Dense(64, activation='swish')(d4)
    output = Dense(64, activation='swish')(d5)
    
    return input, output 

def model2():
    input = Input(shape=(4,))
    d1 = Dense(128, activation='swish')(input)
    d2 = Dense(128, activation='swish')(d1)
    d3 = Dense(128, activation='swish')(d2)
    b1 = BatchNormalization()(d3)
    d4 = Dense(64, activation='swish')(b1)
    d5 = Dense(64, activation='swish')(d4)
    output = Dense(64, activation='swish')(d5)
    
    return input, output 

def model3():
    input = Input(shape=(5,))
    d1 = Dense(128, activation='swish')(input)
    d2 = Dense(128, activation='swish')(d1)
    d3 = Dense(128, activation='swish')(d2)
    b1 = BatchNormalization()(d3)
    d4 = Dense(64, activation='swish')(b1)
    d5 = Dense(64, activation='swish')(d4)
    output = Dense(64, activation='swish')(d5)
    
    return input, output 
    
input1, output1 = model1()
input2, output2 = model2()
input3, output3 = model3()

mg1 = concatenate([output1,output2,output3])
d1 = Dense(32, activation='relu')(mg1)
d2 = Dense(32, activation='relu')(d1)
last_output = Dense(7, activation='softmax')(d2)

model = Model(inputs=[input1,input2,input3],outputs=last_output)

# model = Sequential()
# model.add(Dense(1024, input_shape=(13,),activation='relu'))#, activation='sigmoid'))
# model.add(Dropout(0.05))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.05))
# model.add(Dense(6, activation='relu'))
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.05))
# model.add(Dense(16, activation='relu'))  
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.05))
# model.add(Dense(16, activation='relu'))    
# model.add(Dense(7, activation='softmax'))

#compile & fit
x1_train =np.asarray(x1_train).astype(np.float32) #Numpy는 기본적으로 float32 연산이기 때문에 되도록 맞춰주는게 좋다
x1_test =np.asarray(x1_test).astype(np.float32)
x2_train =np.asarray(x2_train).astype(np.float32) 
x2_test =np.asarray(x2_test).astype(np.float32)
x3_train =np.asarray(x3_train).astype(np.float32) 
x3_test =np.asarray(x3_test).astype(np.float32)
test_csv =np.asarray(test_csv).astype(np.float32)

x1_train = x1_train.reshape(x1_train.shape[0],x1_train.shape[1],1)
x1_test = x1_test.reshape(x1_test.shape[0],x1_test.shape[1],1)
x2_train = x2_train.reshape(x2_train.shape[0],x2_train.shape[1],1)
x2_test = x2_test.reshape(x2_test.shape[0],x2_test.shape[1],1)
x3_train = x3_train.reshape(x3_train.shape[0],x3_train.shape[1],1)
x3_test = x3_test.reshape(x3_test.shape[0],x3_test.shape[1],1)
test_csv = test_csv.reshape(test_csv.shape[0],test_csv.shape[1],1)

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='val_acc',mode='auto',patience=1024,restore_best_weights=True,verbose=1)
# mcp = ModelCheckpoint(monitor='val_loss',mode='min',save_best_only=True,
#                     filepath="c:/_data/_save/MCP/loan/K28_"+"{epoch:04d}{val_loss:.4f}.hdf5")
hist = model.fit([x1_train,x2_train,x3_train], y_train, epochs=16384, batch_size=2048, validation_data=([x1_test,x2_test,x3_test],y_test), verbose=2, callbacks=[es])

#evaluate & predict
# y_test = y_test.reshape(-1)
# print("x_test, y_test: ",x1_test.shape,y_test.shape)

loss = model.evaluate([x1_test,x2_test,x3_test], y_test, verbose=0)    
y_predict = model.predict([x1_test,x2_test,x3_test],verbose=0)
y_predict = np.argmax(y_predict,axis=1)

test_csv1 = test_csv[['대출금액','연간소득','총상환원금','총상환이자']]                         # 수가 크며 분류형이 아닌 데이터들
test_csv2 = test_csv[['최근_2년간_연체_횟수','연체계좌수','부채_대비_소득_비율','총연체금액']]   # 있으면 안좋은 데이터들
test_csv3 = test_csv[['대출기간','근로기간','주택소유상태','총계좌수','대출목적']]
y_submit = np.argmax(model.predict([test_csv1,test_csv2,test_csv3],verbose=0),axis=1)


f1 = f1_score(y_test,y_predict,average='macro')
print(f"{r=}\n LOSS: {loss[0]}\nACC:  {loss[1]}\nF1:   {f1}")

y_submit = label_encoder.inverse_transform(y_submit)

import datetime
dt = datetime.datetime.now()
submission_csv['대출등급'] = y_submit
submission_csv.to_csv(path+f"submit_{dt.day}day{dt.hour:2}{dt.minute:2}_F1{f1:.4f}.csv",index=False)

plt.figure(figsize=(12,9))
plt.title("DACON lClassification")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(hist.history['acc'],label='acc',color='red')
plt.plot(hist.history['val_acc'],label='val_acc',color='blue')
# plt.plot(hist.history['loss'],label='loss',color='red')
# plt.plot(hist.history['val_loss'],label='val_loss',color='blue')
plt.legend()
# plt.show()

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

# r=521
#  LOSS: 0.33680108189582825
# ACC:  0.8795430660247803
# F1:   0.8339690794373145