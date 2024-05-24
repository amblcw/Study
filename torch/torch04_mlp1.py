# torch 1.12.1 cuda 11.4
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#1 data
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]]
             ).T
y = np.array([1,2,3,4,5,6,7,8,9,10])

x = torch.FloatTensor(x)
y = torch.FloatTensor(y)
print(x.shape,y.shape) # torch.Size([3]) torch.Size([3])

y = torch.unsqueeze(y,1)
print(x.shape,y.shape) # torch.Size([3, 1]) torch.Size([3, 1])
# x = x.reshape(-1,1)
# y = y.reshape(-1,1)

print(x,y,sep='\n')

#2 model
model = nn.Sequential(
    nn.Linear(in_features=2,out_features=5),
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
    loss = train(model,criterion,optimizer,x,y)
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
    
evaluate(model,x,y,criterion)

result = model(torch.tensor([[4., 1.3]]).to(device))
print("pred of [4 1.3] =",result.item()) #이것도 먹히지만 이건 결과값이 scalar일때만 가능하다 
# result = torch.Tensor.cpu(result)# item()을 쓰면 이걸 안 해도 된다
# print("pred of [4 1.3]=",result.detach().numpy())
