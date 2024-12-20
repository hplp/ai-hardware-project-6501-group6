import torch
from torch import nn
from net import LeNet
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

normalize = transforms.Normalize([0.1307], [0.3081])

data_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = LeNet().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

def matplot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("train_loss & val_loss")
    plt.show()

def matplot_acc(train_acc, val_acc):
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title("train_acc & val_acc")
    plt.show()

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    loss, current, n = 0.0, 0.0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        output = model(X)
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis=1)

        cur_acc = torch.sum(y == pred)/output.shape[0]

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1

    train_loss = loss / n
    train_acc = current / n
    print("train_loss" + str(train_loss))
    print("train_acc" + str(train_acc))

    return train_loss, train_acc

def val(dataloader, model, loss_fn):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
            val_loss = loss / n
            val_acc = current / n
            print("val_loss" + str(val_loss))
            print("val_acc" + str(val_acc))

            return val_loss, val_acc

epoch = 50
min_acc = 0

loss_train = []
acc_train = []
loss_val = []
acc_val = []

for t in range(epoch):
    print(f'epoch{t+1}\n------------------')
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
    val_loss, val_acc = val(test_dataloader, model, loss_fn)

    loss_train.append(train_loss)
    acc_train.append(train_acc)
    loss_val.append(val_loss)
    acc_val.append(val_acc)

    if val_acc >= min_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir(folder)
        min_acc = val_acc
        print('save best model')
        torch.save(model.state_dict(), folder+'/best_model.pth')

    if t == epoch - 1:
        torch.save(model.state_dict(), folder+'/last_model.pth')

matplot_loss(loss_train, loss_val)
matplot_acc(acc_train, acc_val)
print('Done!')
