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

from sklearn.datasets import load_breast_cancer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#1 data
x, y = load_breast_cancer(return_X_y=True)

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

y_train = torch.unsqueeze(y_train,1)
y_test = torch.unsqueeze(y_test,1)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape) 

from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x_train,y_train)
test_set = TensorDataset(x_test,y_test)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32)

#2 model
class Dnn(nn.Module):
    def __init__(self, input_dim, output_dim, final_activation,hidden_size_list:list=[64,32,16,8]) -> None:
        super(Dnn,self).__init__()
        layer_list = []
        layer_list.append(nn.Linear(input_dim,hidden_size_list[0]))
        for i_size, o_size in zip(hidden_size_list[:-1],hidden_size_list[1:]):
            layer_list.append(nn.Linear(i_size,o_size))
            layer_list.append(nn.ReLU())
        layer_list.append(nn.Linear(hidden_size_list[-1],output_dim))
        layer_list.append(final_activation)
        self.mlp = nn.Sequential(*layer_list).to(device)
        
    def forward(self, x):
        output = self.mlp(x)
        return output

model = Dnn(30,1,nn.Sigmoid())
print(model)

#3 compile & fit
criterion = nn.BCELoss() # criterion: 표준, 기준
optimizer = optim.Adam(model.parameters(),lr=0.1)
optimizer.step()


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
    if i % 100 == 0:
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
