import torch
from torch import nn 
from torch_save_test import NeuralNetwork, device, test_data

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("C:/study/python/torch_model_save/torch_test_model2.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()

x = torch.FloatTensor(test_data[0][0])
x = x.reshape(1,28,28)
y = test_data[0][1]
print(x.shape)

with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
    
