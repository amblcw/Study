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

from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#1 data
x, y = load_digits(return_X_y=True)

y = y.reshape(-1,1)
y = OneHotEncoder(sparse_output=False).fit_transform(y)

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
    nn.Linear(in_features=64,out_features=64),
    nn.SiLU(),
    nn.Linear(64,32),
    nn.BatchNorm1d(32),
    nn.SiLU(),
    nn.Linear(32,16),
    nn.SiLU(),
    nn.Linear(16,8),
    nn.Linear(8,10),
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

    print("pred\n",pred.detach().numpy())
    print("y\n",y.numpy())
    print("loss: ",loss.item())
    acc = accuracy_score(pred,y)
    print("ACC:  ",acc)
    return loss.item()
    
evaluate(model,x_test,y_test,criterion)
