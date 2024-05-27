# torch 1.12.1 cuda 11.4
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import pandas as pd
RANDOM_SEED = 47
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#1 data
path = "C:\\_data\\DACON\\loan\\"

import os
if os.path.exists('torch08_cross_entropy05_x') and os.path.exists('torch08_cross_entropy05_y'):
    with open('torch08_cross_entropy05_x','rb') as x_f:
        x = pickle.load(x_f)
    with open('torch08_cross_entropy05_y','rb') as y_f:
        y = pickle.load(y_f)
else:
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
    x = train_csv.drop(['대출등급'],axis=1).to_numpy()
    y = train_csv['대출등급'].to_numpy()

    print(f"{test_csv.shape}")
    print(np.unique(y,return_counts=True)) #(array([0, 1, 2, 3, 4, 5, 6]), array([16772, 28817, 27622, 13354,  7354,  1954,   420], dtype=int64))



    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    import warnings
    warnings.filterwarnings('ignore')

    y = LabelEncoder().fit_transform(y)
    y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1,1))

    with open('torch08_cross_entropy05_x','+wb') as x_f:
        pickle.dump(x,x_f)
        
    with open('torch08_cross_entropy05_y','+wb') as y_f:
        pickle.dump(y,y_f)
    
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=RANDOM_SEED,stratify=y)
scaler = MinMaxScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape) 

# x_train = torch.unsqueeze(x_train,1)
# x_test = torch.unsqueeze(x_test,1)
# y_train = torch.unsqueeze(y_train,1)
# y_test = torch.unsqueeze(y_test,1)
# print(x_train.shape,y_train.shape,x_test.shape,y_test.shape) 
print(y_train[:5])

#2 model
model = nn.Sequential(
    nn.Linear(in_features=13,out_features=64),
    nn.SiLU(),
    nn.Linear(64,32),
    nn.BatchNorm1d(32),
    nn.SiLU(),
    nn.Linear(32,16),
    nn.SiLU(),
    nn.Linear(16,8),
    nn.Linear(8,7),
    nn.Softmax(),
).to(device)

#3 compile & fit
# model.compile(loss='mse',optimizer='adam') keras 버전
# criterion = nn.MSELoss() # criterion: 표준, 기준
criterion = nn.CrossEntropyLoss() # criterion: 표준, 기준
optimizer = optim.Adam(model.parameters(),lr=0.1)
optimizer.step()

# model.fit(x,y,epoch=100,batch_size=1)

def train(model,criterion,optimizer,x,y):
    model.train()   # 훈련모드, default라서 안해도 상관없음
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()   # 그라디언트 초기화
    hypothesis = model(x)   # 순전파
    hypothesis = hypothesis.to(device)
    loss = criterion(hypothesis,y)  # loss 계산
    loss.backward() # 그라디언트 계산
    optimizer.step()# 가중치 갱신
    return loss.item()  # 이렇게 해야 tensor 형태로 반환됨

EPOCH = 2000
for i in range(1,EPOCH+1):
    loss = train(model,criterion,optimizer,x_train,y_train)
    if i % 100 == 0:
        print(f"epo={i} {loss=:.6f}")
else:
    print("======= train finish =======")

# predict
def evaluate(model, x, y, criterion):
    model.eval()
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        pred = model(x)
        pred = pred.to(device)
        loss = criterion(pred,y)
    pred = torch.Tensor.cpu(pred)
    y = torch.Tensor.cpu(y)
    
    from sklearn.metrics import accuracy_score
    pred = np.argmax(pred.squeeze(),axis=1)
    y = np.argmax(y.squeeze(),axis=1)

    # print("pred\n",pred.detach().numpy())
    # print("y\n",y.numpy())
    print("loss: ",loss.item())
    acc = accuracy_score(pred,y)
    print("ACC:  ",acc)
    return loss.item()
    
evaluate(model,x_test,y_test,criterion)
# loss:  1.348068356513977
# ACC:   0.817225028613719