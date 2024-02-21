import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from keras.datasets import fashion_mnist
import os
import pandas as pd
from torchvision.io import read_image
print(torch.__version__)    # 2.2.0+cu118

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(y_train[0],y_test[0])

class CustomImageDataset(Dataset):
    def __init__(self,x_data,y_data,transform=None) -> None:
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform
        
    def __len__(self):
        return len(self.y_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = torch.FloatTensor(self.x_data[idx].copy())
        label = self.y_data[idx].copy()
        sample = image, label
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

import matplotlib.pyplot as plt

# training_data = datasets.FashionMNIST(
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

# print(type(f_mnist))
# print(type(training_data))

training_data = CustomImageDataset(x_train,y_train)
test_data = CustomImageDataset(x_test, y_test)

BATCH_SIZE = 256

train_dataloader = DataLoader(training_data,batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data,batch_size=BATCH_SIZE)

for X, y in test_dataloader:
    # print("===================")
    # print(X,y,sep='\n')
    print("-------------------")
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

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # 예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1)*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
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