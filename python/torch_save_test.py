import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from keras.datasets import fashion_mnist
import os
import pandas as pd
import numpy as np
from torchvision.io import read_image
print(torch.__version__)    # 2.2.0+cu118

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]]).reshape(-1,3,1)
y = np.array([4,5,6,7,8,9,10])
print(x.shape,y.shape) # (7, 3, 1) (7,)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(y_train[0],y_test[0])

class CustomImageDataset(Dataset):
    def __init__(self,x_data,y_data,transform=None) -> None:    # 생성자, x,y데이터, 변환함수 설정
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform
        
    def __len__(self):          # 어떤 값을 길이로서 반환할지
        return len(self.y_data)
    
    def __getitem__(self, idx): # 인덱스가 들어왔을 때 어떻게 값을 반환할지 
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = torch.FloatTensor(self.x_data[idx].copy())  # torchTensor 형식으로 변환
        label = self.y_data[idx].copy()                     # 얘는 9를 변환하면 사이즈9짜리 텐서로 만들어버리기에 그냥 이대로 사용
        sample = image, label                               # 반드시 x, y 순으로 반환할 것
        
        if self.transform:  # 전처리 함수 적용
            sample = self.transform(sample)
            
        return sample

import matplotlib.pyplot as plt

# training_data = datasets.FashionMNIST(    # 기존 데이터셋 불러오는 방식
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor()
# )

# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor()
# )

training_data = CustomImageDataset(x_train,y_train) # 인스턴스 선언 및 데이터 지정
test_data = CustomImageDataset(x_test, y_test)

BATCH_SIZE = 100
train_dataloader = DataLoader(training_data,batch_size=BATCH_SIZE)  # 만든 커스텀 데이터를 iterator형식으로 변경
test_dataloader = DataLoader(test_data,batch_size=BATCH_SIZE)

for X, y in test_dataloader:    # dataloader는 인덱스로 접근이 되지 않으며 .next()또한 사용할 수 없다 오직 for문만 가능하며 그렇기에 출력을 위해 한바퀴 돌자마자 break한다
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shaep of y: {y.shape} {y.dtype}")
    break


device = (
    "cuda"
    if torch.cuda.is_available() 
    else "mps" 
    if torch.backends.mps.is_available() 
    else "cpu"
)

class NeuralNetwork(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,10)
        )
        
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss() # 이 함수에 softmax가 내재되어있기에 모델에서 softmax를 쓰면 안된다
optimizer = torch.optim.Adam(model.parameters())

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # 예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # 역전파
        optimizer.zero_grad()   # 역전파하기 전에 기울기를 0으로 만들지 않으면 전의 학습이 영향을 준다
        loss.backward()         # 역전파 계산
        optimizer.step()        # 역전파 계산에 따라 파라미터 업데이트(이 시점에서 가중치가 업데이트 된다)
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1)*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # print("size check: ",size,num_batches) # size는 테스트 데이터의 개수, num_batches는 batch_size에 의해 몇 바퀴 도는지
    model.eval()    # 모델을 평가 모드로 전환, Dropout이나 BatchNomalization등을 비활성화
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() 
            # 예측값을 argmax로 확실한 정수로 만들고 정답과 비교해서 참 거짓을 만든다, 그리고 float으로 만든뒤 다 더해서 최종적으로 모든 데이터에서 정답값의 개수를 얻게된다
    test_loss /= num_batches    # 그렇게 얻은 총 정답개수를 전체 데이터 개수로 다 나누면 그게 곧 acc가 된다
    correct /= size
    print(f"Test Error \n Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f}\n")
    
if __name__ == '__main__':
    print(f"Using {device} device")
    print(model)
    
    EPOCHS = 50
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n---------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done")

    import os
    dir_path = os.getcwd()
    print(dir_path)
    torch.save(model.state_dict(), dir_path+"./python/torch_model_save/torch_test_model2.pth")
    print("Model saved")