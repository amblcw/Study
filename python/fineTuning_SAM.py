from segment_anything import SamPredictor, sam_model_registry
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import rasterio
import torch
import os

# Loading the model based on checkpoint
SAM_CHECKPOINT = "c:/_data/SAM/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
MAX_PIXEL_VALUE = 65535 # 이미지 정규화를 위한 픽셀 최대값
PATH = 'c:/aifactory/'

sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
predictor = SamPredictor(sam)

# Define dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, BAND, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(self.root_dir)
        self.BAND = BAND

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 원본코드
        # img_name = os.path.join(self.root_dir, self.images[idx])
        # image = Image.open(img_name)
        
        img = rasterio.open(self.root_dir  + f'train_img/train_img_{idx}.tif').read(self.BAND).transpose((1, 2, 0))
        mask = rasterio.open(self.root_dir  + f'test_img/test_img_{idx}.tif').read(1).astype(np.float32) / MAX_PIXEL_VALUE
        
        img = img.astype(np.float32)
        # print("img shape",img.shape)
        # img = torch.Tensor(img)
        # mask = torch.Tensor(mask)
        # print("img shape",img.shape)
        

        # if self.transform:
        #     img = self.transform(img)
        #     mask = self.transform(mask)


        return img, mask

# Define transforms
transform = transforms.Compose([
    transforms.Resize((1024, 1024)), # Resize to the size the model expects
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalization values for pre-trained PyTorch models
])

# Load custom dataset
dataset = CustomDataset(root_dir=PATH, transform=transform, BAND=(7,6,8))

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Fine-tuning the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
predictor.model.to(device)
predictor.model.train()

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(predictor.model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, (x,y) in enumerate(dataloader):
        x.to(device)
        y.to(device)
        print(x.shape)
        x = x.reshape(256,256,3)
        
        predictor.set_image(x)
        outputs = predictor.model(x, True)
        loss = criterion(outputs, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')