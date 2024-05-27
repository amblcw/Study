# torch 1.12.1 cuda 11.4
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#1 data
x_train = np.array([1,2,3,4,5,6,7,])
y_train = np.array([1,2,3,4,5,6,7,])
x_test = np.array([1,2,3,4,5,6,7,])
y_test = np.array([1,2,3,4,5,6,7,])

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape) 

x_train = torch.unsqueeze(x_train,1)
x_test = torch.unsqueeze(x_test,1)
y_train = torch.unsqueeze(y_train,1)
y_test = torch.unsqueeze(y_test,1)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape) 

#2 model
model = nn.Sequential(
    nn.Linear(in_features=1,out_features=5),
    nn.Linear(5,4),
    nn.Linear(4,3),
    nn.Linear(3,2),
    nn.Linear(2,1),
).to(device)

#3 compile & fit
# model.compile(loss='mse',optimizer='adam') keras 버전
criterion = nn.MSELoss() # criterion: 표준, 기준
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
    print("pred\n",pred.detach().numpy())
    print("y\n",y.numpy())
    print("loss: ",loss.item())
    return loss.item()
    
evaluate(model,x_test,y_test,criterion)

result = model(torch.tensor([[4.]]).to(device))
print("pred of 4 =",result.item()) #이것도 먹히지만 이건 결과값이 scalar일때만 가능하다 
# result = torch.Tensor.cpu(result)# item()을 쓰면 이걸 안 해도 된다
# print("pred of 4 =",result.detach().numpy())
