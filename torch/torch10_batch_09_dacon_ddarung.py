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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#1 data
path = "C:\\_data\\DACON\\ddarung\\"
train_csv = pd.read_csv(path+"train.csv",index_col=['id'])  
test_csv = pd.read_csv(path+"test.csv",index_col=0)         
submission_csv = pd.read_csv(path+"submission.csv")

train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = np.asarray(train_csv.drop(['count'],axis=1)) #count 를 드랍, axis=0은 행, axis=1은 열
y = np.asarray(train_csv['count'])

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=RANDOM_SEED)
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
y_train = torch.unsqueeze(y_train,1)
y_test = torch.unsqueeze(y_test,1)
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

model = Dnn(9,1)
print(model)

#3 compile & fit
# model.compile(loss='mse',optimizer='adam') keras 버전
# criterion = nn.MSELoss() # criterion: 표준, 기준
criterion = nn.MSELoss() # criterion: 표준, 기준
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
        print(pred.shape,y.shape)
    total_loss /= len(data_loader)
    
    predict = np.vstack(predict)
    y_true = np.vstack(y_true)
    
    from sklearn.metrics import r2_score
    r2 = r2_score(predict,y_true)
    
    print("loss: ",total_loss)
    print("R2  : ",r2)
    return total_loss
    
evaluate(model,test_loader,criterion)
