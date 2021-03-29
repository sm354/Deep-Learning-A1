import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os
import time
import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np
from model import *
from dataset import *
from torch.utils.data import Dataset, DataLoader

def main():
    exp_name='ResNetA'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(torch.cuda.get_device_properties(0))
    
    # Data
    print('==> Preparing data..')
    train_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), 
    ])

    trainset = Cifar10(root='../CIFAR10/',train=True,transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = Cifar10(root='../CIFAR10/',train=False,transform=test_transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # Model
    print('==> Building model..')
    net = ResNetModel(n=2,r=10)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    start_epoch = 1
    end_epoch = 76
    best_acc=0

    # Load checkpoint.
    if os.path.exists('%s_model.pth'%exp_name):        
        print('==> Resuming from last epoch..')
        checkpoint = torch.load('%s_model.pth'%exp_name)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    lr=0.1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    trainloss, testloss, trainacc, testacc = [], [], [], []
    for epoch in range(start_epoch, end_epoch):
        print('epoch',epoch)
        if epoch==50:
            lr=lr/10.
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)

        l,a=train(epoch, net, trainloader, device, optimizer, criterion)
        trainloss.append(l)
        trainacc.append(a)
        l,a,best_acc=test(epoch, best_acc, net, testloader, device, criterion, exp_name)
        testloss.append(l)
        testacc.append(a)

        print(trainloss[-1], trainacc[-1])
        print(testloss[-1], testacc[-1])

    state = {
        'net': net.state_dict(),
        'acc': testacc[-1],
        'epoch': epoch,
    }
    torch.save(state, '%s_model.pth'%exp_name)
    Save_Stats(trainloss, trainacc, testloss, testacc, exp_name)

def train(epoch, net, trainloader, device, optimizer, criterion):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return train_loss/(batch_idx+1), 100.*correct/total 

def test(epoch, best_acc, net, testloader, device, criterion,exp_name):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, '%s_checkpoint.pth'%exp_name)
        best_acc = acc
    return test_loss/(batch_idx+1), acc, best_acc

def Save_Stats(trainloss, trainacc, testloss, testacc, exp_name):
    data=[]
    data.append(trainloss)
    data.append(testloss)
    data.append(trainacc)
    data.append(testacc)
    data=np.array(data)
    data.reshape((4,-1))
    stats_path = '%s.npy'%exp_name
    if os.path.exists(stats_path):
        data_old = np.load(stats_path)
        data = np.hstack((data_old,data))
    np.save(stats_path,data)

    SavePlots(data[0], data[1], 'loss', exp_name)
    SavePlots(data[2], data[3], 'acc', exp_name)

def SavePlots(y1, y2, metric, exp_name):
    try:
        plt.clf()
    except Exception as e:
        pass
    # plt.style.use('seaborn')
    plt.title(exp_name)
    plt.xlabel('epochs')
    plt.ylabel(metric)
    epochs=np.arange(1,len(y1)+1,1)
    plt.plot(epochs,y1,label='train %s'%metric)
    plt.plot(epochs,y2,label='val %s'%metric)
    if metric=='acc':
        ep=np.argmax(y2)
        plt.plot(ep+1,y2[ep],'r*',label='bestacc@(%i,%i)'%(ep+1,y2[ep]))
    plt.legend()
    plt.savefig('%s_%s'%(exp_name,metric), dpi=95)

if __name__ == '__main__':
    main()
