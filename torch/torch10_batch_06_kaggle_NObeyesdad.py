# torch 1.12.1 cuda 11.4
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
RANDOM_SEED = 47
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#1 data
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

path = "C:\_data\KAGGLE\playground-series-s4e2\\"
train_csv = pd.read_csv(path+"train.csv", index_col=0)
test_csv = pd.read_csv(path+"test.csv", index_col=0)

# print(train_csv.shape, test_csv.shape, submit_csv.shape)    # (20758, 17) (13840, 16) (13840, 2)
# for label in train_csv:
#         print(train_csv[label].isna().sum())    # 결측치 없음을 확인

class_label = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS','NObeyesdad']

''' # train csv 라벨들 확인
for label in train_csv:
        if label in class_label:
                print(label ,np.unique(train_csv[label], return_counts=True))
Gender (array(['Female', 'Male'], dtype=object), array([10422, 10336], dtype=int64))
family_history_with_overweight (array(['no', 'yes'], dtype=object), array([ 3744, 17014], dtype=int64))
FAVC (array(['no', 'yes'], dtype=object), array([ 1776, 18982], dtype=int64))
CAEC (array(['Always', 'Frequently', 'Sometimes', 'no'], dtype=object), array([  478,  2472, 17529,   279], dtype=int64))
SMOKE (array(['no', 'yes'], dtype=object), array([20513,   245], dtype=int64))
SCC (array(['no', 'yes'], dtype=object), array([20071,   687], dtype=int64))
CALC (array(['Frequently', 'Sometimes', 'no'], dtype=object), array([  529, 15066,  5163], dtype=int64))
MTRANS (array(['Automobile', 'Bike', 'Motorbike', 'Public_Transportation',
       'Walking'], dtype=object), array([ 3534,    32,    38, 16687,   467], dtype=int64))
NObeyesdad (array(['Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I',
       'Obesity_Type_II', 'Obesity_Type_III', 'Overweight_Level_I',
       'Overweight_Level_II'], dtype=object), array([2523, 3082, 2910, 3248, 4046, 2427, 2522], dtype=int64))
'''
''' # test csv 라벨들 확인
for label in test_csv:
        if label in class_label:
                print(label, np.unique(test_csv[label], return_counts=True))
Gender (array(['Female', 'Male'], dtype=object), array([6965, 6875], dtype=int64))
family_history_with_overweight (array(['no', 'yes'], dtype=object), array([ 2456, 11384], dtype=int64))
FAVC (array(['no', 'yes'], dtype=object), array([ 1257, 12583], dtype=int64))
CAEC (array(['Always', 'Frequently', 'Sometimes', 'no'], dtype=object), array([  359,  1617, 11689,   175], dtype=int64))
SMOKE (array(['no', 'yes'], dtype=object), array([13660,   180], dtype=int64))
SCC (array(['no', 'yes'], dtype=object), array([13376,   464], dtype=int64))
CALC (array(['Always', 'Frequently', 'Sometimes', 'no'], dtype=object), array([   2,  346, 9979, 3513], dtype=int64))
MTRANS (array(['Automobile', 'Bike', 'Motorbike', 'Public_Transportation',
       'Walking'], dtype=object), array([ 2405,    25,    19, 11111,   280], dtype=int64))
'''
test_csv.loc[test_csv['CALC'] == 'Always', 'CALC'] = 'Frequently'
# print(train_csv.head)
''' # train head 확인
<bound method NDFrame.head of        
        Gender      Age    Height      Weight   family_history_with_overweight FAVC      FCVC     NCP        CAEC   SMOKE    CH2O   SCC    FAF       TUE       CALC                 MTRANS           NObeyesdad
id
0        Male  24.443011  1.699998   81.669950                            yes  yes  2.000000  2.983297   Sometimes    no  2.763573  no  0.000000  0.976473  Sometimes  Public_Transportation  Overweight_Level_II
1      Female  18.000000  1.560000   57.000000                            yes  yes  2.000000  3.000000  Frequently    no  2.000000  no  1.000000  1.000000         no             Automobile        Normal_Weight
2      Female  18.000000  1.711460   50.165754                            yes  yes  1.880534  1.411685   Sometimes    no  1.910378  no  0.866045  1.673584         no  Public_Transportation  Insufficient_Weight
3      Female  20.952737  1.710730  131.274851                            yes  yes  3.000000  3.000000   Sometimes    no  1.674061  no  1.467863  0.780199  Sometimes  Public_Transportation     Obesity_Type_III
4        Male  31.641081  1.914186   93.798055                            yes  yes  2.679664  1.971472   Sometimes    no  1.979848  no  1.967973  0.931721  Sometimes  Public_Transportation  Overweight_Level_II
...       ...        ...       ...         ...                            ...  ...       ...       ...         ...   ...       ...  ..       ...       ...        ...                    ...                  ...
20753    Male  25.137087  1.766626  114.187096                            yes  yes  2.919584  3.000000   Sometimes    no  2.151809  no  1.330519  0.196680  Sometimes  Public_Transportation      Obesity_Type_II
20754    Male  18.000000  1.710000   50.000000                             no  yes  3.000000  4.000000  Frequently    no  1.000000  no  2.000000  1.000000  Sometimes  Public_Transportation  Insufficient_Weight
20755    Male  20.101026  1.819557  105.580491                            yes  yes  2.407817  3.000000   Sometimes    no  2.000000  no  1.158040  1.198439         no  Public_Transportation      Obesity_Type_II
20756    Male  33.852953  1.700000   83.520113                            yes  yes  2.671238  1.971472   Sometimes    no  2.144838  no  0.000000  0.973834         no             Automobile  Overweight_Level_II
20757    Male  26.680376  1.816547  118.134898                            yes  yes  3.000000  3.000000   Sometimes    no  2.003563  no  0.684487  0.713823  Sometimes  Public_Transportation      Obesity_Type_II
'''
# print(train_csv.columns)
# ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
#        'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
#        'CALC', 'MTRANS', 'NObeyesdad']
x_labelEncoder = LabelEncoder()
train_csv['Gender'] = x_labelEncoder.fit_transform(train_csv['Gender'])
train_csv['family_history_with_overweight'] = x_labelEncoder.fit_transform(train_csv['family_history_with_overweight'])
train_csv['FAVC'] = x_labelEncoder.fit_transform(train_csv['FAVC'])
train_csv['CAEC'] = x_labelEncoder.fit_transform(train_csv['CAEC'])
train_csv['SMOKE'] = x_labelEncoder.fit_transform(train_csv['SMOKE'])
train_csv['SCC'] = x_labelEncoder.fit_transform(train_csv['SCC'])
train_csv['CALC'] = x_labelEncoder.fit_transform(train_csv['CALC'])
train_csv['MTRANS'] = x_labelEncoder.fit_transform(train_csv['MTRANS'])

x_labelEncoder = LabelEncoder()
test_csv['Gender'] = x_labelEncoder.fit_transform(test_csv['Gender'])
test_csv['family_history_with_overweight'] = x_labelEncoder.fit_transform(test_csv['family_history_with_overweight'])
test_csv['FAVC'] = x_labelEncoder.fit_transform(test_csv['FAVC'])
test_csv['CAEC'] = x_labelEncoder.fit_transform(test_csv['CAEC'])
test_csv['SMOKE'] = x_labelEncoder.fit_transform(test_csv['SMOKE'])
test_csv['SCC'] = x_labelEncoder.fit_transform(test_csv['SCC'])
test_csv['CALC'] = x_labelEncoder.fit_transform(test_csv['CALC'])
test_csv['MTRANS'] = x_labelEncoder.fit_transform(test_csv['MTRANS'])

y_labelEncoder = LabelEncoder()
train_csv['NObeyesdad'] = y_labelEncoder.fit_transform(train_csv['NObeyesdad'])
# print(train_csv.head)
''' # 라벨 인코딩 후 train.head 확인
<bound method NDFrame.head of        
       Gender     Age      Height     Weight   family_history_with_overweight    FAVC    CVC       NCP     CAEC  SMOKE   CH2O    SCC    FAF       TUE      CALC   MTRANS    NObeyesdad
id
0           1  24.443011  1.699998   81.669950                               1     1  2.000000  2.983297     2      0  2.763573    0  0.000000  0.976473     1       3           6
1           0  18.000000  1.560000   57.000000                               1     1  2.000000  3.000000     1      0  2.000000    0  1.000000  1.000000     2       0           1
2           0  18.000000  1.711460   50.165754                               1     1  1.880534  1.411685     2      0  1.910378    0  0.866045  1.673584     2       3           0
3           0  20.952737  1.710730  131.274851                               1     1  3.000000  3.000000     2      0  1.674061    0  1.467863  0.780199     1       3           4
4           1  31.641081  1.914186   93.798055                               1     1  2.679664  1.971472     2      0  1.979848    0  1.967973  0.931721     1       3           6
...       ...        ...       ...         ...                             ...   ...       ...       ...   ...    ...       ...  ...       ...       ...   ...     ...         ...
20753       1  25.137087  1.766626  114.187096                               1     1  2.919584  3.000000     2      0  2.151809    0  1.330519  0.196680     1       3           3
20754       1  18.000000  1.710000   50.000000                               0     1  3.000000  4.000000     1      0  1.000000    0  2.000000  1.000000     1       3           0
20755       1  20.101026  1.819557  105.580491                               1     1  2.407817  3.000000     2      0  2.000000    0  1.158040  1.198439     2       3           3
20756       1  33.852953  1.700000   83.520113                               1     1  2.671238  1.971472     2      0  2.144838    0  0.000000  0.973834     2       0           6
20757       1  26.680376  1.816547  118.134898                               1     1  3.000000  3.000000     2      0  2.003563    0  0.684487  0.713823     1       3           3
'''


""" # P 검정
import scipy.stats as stats
for label in train_csv:
    print(label," ",stats.pearsonr(train_csv['NObeyesdad'],train_csv[label]))
Gender   PearsonRResult(statistic=0.046574912033978184, pvalue=1.8990220650683642e-11)
Age   PearsonRResult(statistic=0.2830183712239907, pvalue=0.0)
Height   PearsonRResult(statistic=0.060785550400480136, pvalue=1.8621467518362792e-18)
Weight   PearsonRResult(statistic=0.43182097207728903, pvalue=0.0)
family_history_with_overweight   PearsonRResult(statistic=0.32132484319938587, pvalue=0.0)
        FAVC   PearsonRResult(statistic=0.010176246176913503, pvalue=0.14261934009881228)
FCVC   PearsonRResult(statistic=0.0410763864808035, pvalue=3.2139644150763383e-09)
NCP   PearsonRResult(statistic=-0.09115416942846906, pvalue=1.4953264119959517e-39)
CAEC   PearsonRResult(statistic=0.297419757072015, pvalue=0.0)
        SMOKE   PearsonRResult(statistic=-0.0013927633524938529, pvalue=0.84097046043243)
CH2O   PearsonRResult(statistic=0.18709958001025073, pvalue=7.344610346043163e-163)
SCC   PearsonRResult(statistic=-0.06517135385355988, pvalue=5.504265380370881e-21)
FAF   PearsonRResult(statistic=-0.09664292513984239, pvalue=2.9003365419591887e-44)
TUE   PearsonRResult(statistic=-0.07603955730067295, pvalue=5.284478962295593e-28)
CALC   PearsonRResult(statistic=-0.16849742485140495, pvalue=5.0277297570113e-132)
MTRANS   PearsonRResult(statistic=-0.07743006081693204, pvalue=5.594245339542871e-29)
NObeyesdad   PearsonRResult(statistic=1.0, pvalue=0.0) """

''' BMI 컬럼 추가 '''
train_csv['BMI'] = train_csv['Weight'] / (train_csv['Height']*train_csv['Height'])
test_csv['BMI'] = test_csv['Weight'] / (test_csv['Height']*test_csv['Height'])

''' 이상치 제거 '''
age_q1 = train_csv['Age'].quantile(0.25)
age_q3 = train_csv['Age'].quantile(0.75)
age_gap = (age_q3 - age_q1 ) * 1.5
age_under = age_q1 - age_gap
age_upper = age_q3 + age_gap
train_csv = train_csv[train_csv['Age']>=age_under]
train_csv = train_csv[train_csv['Age']<=age_upper]

weight_q1 = train_csv['Weight'].quantile(0.25)
weight_q3 = train_csv['Weight'].quantile(0.75)
weight_gap = (weight_q3 - weight_q1 ) * 1.5
weight_under = weight_q1 - weight_gap
weight_upper = weight_q3 + weight_gap
train_csv = train_csv[train_csv['Weight']>=weight_under]
train_csv = train_csv[train_csv['Weight']<=weight_upper]

x = train_csv.drop(['NObeyesdad'], axis=1) # P검정에 의거하여 FAVC와 SMOKE 제거
y = train_csv['NObeyesdad']

'''# 최대 최소 1분위 3분위 구하기
for label in x:         
        if label in class_label:
                continue
        print(f"{label:30}: max={max(x[label]):<10}  min={min(x[label]):<10}  q1={x[label].quantile(0.25):<10}  q3={x[label].quantile(0.75):<10}")
Age                           : max=61.0        min=14.0        q1=20.0        q3=26.0
Height                        : max=1.975663    min=1.45        q1=1.631856    q3=1.762887
Weight                        : max=165.057269  min=39.0        q1=66.0        q3=111.600553
FCVC                          : max=3.0         min=1.0         q1=2.0         q3=3.0
NCP                           : max=4.0         min=1.0         q1=3.0         q3=3.0
CH2O                          : max=3.0         min=1.0         q1=1.792022    q3=2.549617
FAF                           : max=3.0         min=0.0         q1=0.008013    q3=1.587406
TUE                           : max=2.0         min=0.0         q1=0.0         q3=1.0
'''

# model 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
y = LabelEncoder().fit_transform(y)
# y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1,1))
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, train_size=0.8,
    stratify=y
)

print(x_train.shape,y_train.shape)
print(np.unique(y_train,return_counts=True))


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape) 

from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x_train,y_train)
test_set = TensorDataset(x_test,y_test)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64)

#2 model
class Dnn(nn.Module):
    def __init__(self, input_dim, output_dim, final_activation=None,hidden_size_list:list=[64,32,16,8]) -> None:
        super(Dnn,self).__init__()
        layer_list = []
        layer_list.append(nn.Linear(input_dim,hidden_size_list[0]))
        for i_size, o_size in zip(hidden_size_list[:-1],hidden_size_list[1:]):
            layer_list.append(nn.Linear(i_size,o_size))
            layer_list.append(nn.SiLU())
        layer_list.append(nn.Linear(hidden_size_list[-1],output_dim))
        if final_activation is not None:
            layer_list.append(final_activation)
        self.mlp = nn.Sequential(*layer_list).to(device)
        
    def forward(self, x):
        output = self.mlp(x)
        return output

model = Dnn(17,7)
print(model)

#3 compile & fit
# model.compile(loss='mse',optimizer='adam') keras 버전
# criterion = nn.MSELoss() # criterion: 표준, 기준
criterion = nn.CrossEntropyLoss() # criterion: 표준, 기준
optimizer = optim.Adam(model.parameters(),lr=0.1)
optimizer.step()

# model.fit(x,y,epoch=100,batch_size=1)

def train(model,criterion,optimizer,data_loader):
    model.train()   # 훈련모드, default라서 안해도 상관없음
    total_loss = 0
    for x_batch, y_batch in data_loader:
        x, y = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()   # 그라디언트 초기화
        hypothesis = model(x)   # 순전파
        loss = criterion(hypothesis,y)  # loss 계산
        loss.backward() # 그라디언트 계산
        optimizer.step()# 가중치 갱신
        total_loss += loss.item()
    total_loss = total_loss / len(data_loader) 
    return total_loss  # 이렇게 해야 tensor 형태로 반환됨

EPOCH = 2000
PATIENCE = 200
best_loss = 987654321
patience = PATIENCE
for i in range(1,EPOCH+1):
    if patience <= 0:
        print("train stopped at",i,"epo")
        break
    loss = train(model,criterion,optimizer,train_loader)
    if loss < best_loss:
        best_loss = loss
        patience = PATIENCE
    if i % 10 == 0:
        print(f"epo={i} {loss=:.6f}")
    patience -= 1
print("======= train finish =======")

# predict
def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    predict = []
    y_true = []
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            total_loss += criterion(pred,y).item()
        predict.append(pred.cpu().detach().numpy())
        y_true.append(y.cpu().numpy())
    total_loss /= len(data_loader)
    
    predict = np.vstack(predict)
    y_true = np.vstack(y_true)
    print("predict, y_true shape",predict.shape,y_true.shape)
    
    from sklearn.metrics import accuracy_score
    predict = np.round(predict.squeeze())
    y_true = y_true.squeeze()
    acc = accuracy_score(predict,y_true)
    
    # print("pred\n",pred.detach().numpy())
    # print("y\n",y.numpy())
    print("loss: ",total_loss)
    print("ACC:  ",acc)
    return total_loss
    
evaluate(model,test_loader,criterion)