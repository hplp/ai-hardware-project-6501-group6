import torch
from torch import nn
from net_quant import LeNet
import time
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
from  torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def read_8bit_img(filepath):
    image = Image.open(filepath).convert('L')
    resize = transforms.Resize([28, 28])
    image = resize(image)
    image = np.copy(image)
    image = torch.tensor(image)
    image = Variable(torch.unsqueeze(torch.unsqueeze(image, dim=0).int(), dim=0).int()).to(device)
    image = image.clone().detach().to(device)
    return image

def read_float_img(filepath):
    # ROOT_TEST = r'D:/ws_pytorch/LeNet5/data/mydata'
    # test_transform = transforms.Compose([
    #     transforms.Grayscale(num_output_channels=1),
    #     transforms.Resize((28, 28)),
    #     transforms.ToTensor()])
    # test_dataset = ImageFolder(ROOT_TEST, transform=test_transform)
    # image = test_dataset[0][0]
    # image = Variable(torch.unsqueeze(image, dim=0).float(), requires_grad=True).to(device)
    # image = image.clone().detach().to(device)

    image = Image.open(filepath).convert('L')
    resize = transforms.Resize([28, 28])
    image = resize(image)
    image = np.copy(image)
    image = torch.tensor(image)
    image = Variable(torch.unsqueeze(torch.unsqueeze(image, dim=0).float(), dim=0).float()).to(device)
    image = image.clone().detach().to(device)
    return image

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

model1 = LeNet().to(device)
model1.load_state_dict(torch.load("./save_model/best_model.pth"))

model2 = LeNet().to(device)
model2.load_state_dict(torch.load("./save_model/quant_model.pth"))
model2.load_quant(25, 12, 80, 131, 15, 140, 23, 14, 124, 13, 13, 131, 51, 14, 127)

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

model1.eval()
model2.eval()

print("Origial model test:")
correct = 0
total = 0
start1 = time.time()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        pred = model1(images)
        predicted = torch.argmax(pred, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
accuracy1 = correct / total * 100
end1 = time.time()
print(f"Accuracy: {accuracy1:.2f}%")
print("#" * 20)

print("Quantized int8 model test:")
correct = 0
total = 0
start2 = time.time()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = (images * 255).int().to(device), labels.to(device)  # 模拟量化的8位图像输入
        pred = model2(images)
        predicted = torch.argmax(pred, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
accuracy2 = correct / total * 100
end2 = time.time()
print(f"Accuracy: {accuracy2:.2f}%")
