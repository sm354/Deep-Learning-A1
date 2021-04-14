import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import time
import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np
import argparse
from ResNets import *
from Normalisations import *

# IMPORTANT NOTE - 'cifar-10-batches-py' this should not be in data_dir but submission instructions has it! Either delete string or shout on piazza
def split_cifar(data_dir): 
    t = transforms.Compose([transforms.ToTensor(),])
    ts = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=t)
    loader = DataLoader(ts, batch_size=40000, shuffle=True, num_workers=2)
    a,b=[],[]
    for _, (X,Y) in enumerate(loader):
        a.append(X)
        b.append(Y)
    return a[0], b[0], a[1], b[1], torch.mean(torch.cat((a[0],a[1]),dim=0),dim=0) # Xtrain, Ytrain, Xval, Yval, per_pixel_mean (3,32,32)

class Cifar10(Dataset):
    def __init__(self, X, Y, transform=None):
        super(Cifar10, self).__init__()
        self.transform=transform
        self.X=X
        self.Y=Y
    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = self.transform(x)
        return x, y
    def __len__(self):
        return self.X.shape[0]


def Save_Stats(trainacc, testacc, exp_name):
    data=[]
    data.append(trainacc)
    data.append(testacc)
    data=np.array(data)
    data.reshape((2,-1))

    if not os.path.exists('./ForReport'):
        os.mkdir('./ForReport')
    stats_path = './ForReport/%s_accs.npy'%exp_name
    # if os.path.exists(stats_path):
    #     data_old = np.load(stats_path)
    #     data = np.hstack((data_old,data))
    np.save(stats_path,data)
    SavePlots(data[0], data[1], 'acc', exp_name)

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
    ep=np.argmax(y2)
    plt.plot(ep+1,y2[ep],'r*',label='bestacc@(%i,%i)'%(ep+1,y2[ep]))
    plt.legend()
    plt.savefig('./ForReport/%s_%s'%(exp_name,metric), dpi=95)

parser = argparse.ArgumentParser(description='Training ResNets with Normalizations on CIFAR10')
parser.add_argument('--normalization', type=str)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--output_file', type=str)
parser.add_argument('--n', help='number of (per) residual blocks', type=int)
parser.add_argument('--r', default=10, help='number of classes', type=int)
args=parser.parse_args()

n=args.n
r=args.r
normalization_layers = {'torch_bn':nn.BatchNorm2d,'bn':BatchNorm2D,'in':InstanceNorm2D,'bin':BIN2D,'ln':LayerNorm2D,'gn':GroupNorm2D,'nn':None}
norm_layer_name=args.normalization
norm_layer=normalization_layers[norm_layer_name]

# create the required ResNet model
net = ResNet(n,r,norm_layer_name,norm_layer)
# print(net(torch.rand((2,3,32,32))).shape) : all resnet models working [CHECKED]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# create train-val-test split of CIFAR10
X_train, Y_train, X_val, Y_val, per_pixel_mean = split_cifar(data_dir=args.data_dir) #torch.tensorsxw
# print(per_pixel_mean.shape)
train_transform = transforms.Compose(
    [transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(1.,1.,1.)), #per_pixel_mean : HOLD ERROR : RuntimeError: output with shape [3, 32, 32] doesn't match the broadcast shape [3, 1, 3, 32, 32]
]) # data augmentation as per the paper. Note the normalisation is done in dataset.py script
val_transform = transforms.Compose(
    [transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(1.,1.,1.)), #per_pixel_mean : HOLD ERROR : RuntimeError: output with shape [3, 32, 32] doesn't match the broadcast shape [3, 1, 3, 32, 32]
])
trainset = Cifar10(X_train, Y_train, transform=train_transform)
trainset_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
valset = Cifar10(X_val, Y_val, transform=val_transform)
valset_loader = DataLoader(valset, batch_size=128, shuffle=False, num_workers=2)

############
#     if os.path.exists(PATH):        
#         net.load_state_dict(torch.load(PATH))
# #############

# Begin training
start_epoch, end_epoch = 1, 3
loss_fn = nn.CrossEntropyLoss()
lr=0.1
optimizer = optim.SGD(net.parameters(),lr=lr)
train_acc, val_acc = [], []
for epoch in range(start_epoch, end_epoch):
    print(epoch)
    # Training 
    net.train()
    total_samples, correct_predictions = 0, 0
    for _, (X,Y) in enumerate(trainset_loader):
        X=X.to(device)
        Y=Y.to(device)
        optimizer.zero_grad() # remove history
        Y_ = net(X)
        loss = loss_fn(Y_, Y)
        loss.backward() # create computational graph i.e. find gradients
        optimizer.step() # update weights/biases
        __, Y_predicted = Y_.max(1)
        total_samples += Y_predicted.size(0)
        correct_prediction = Y_predicted.eq(Y).sum().item()
        correct_predictions += correct_prediction
    train_acc.append( (correct_predictions/total_samples)*100. )

    # Testing
    net.eval() #this is useful in informing nn.modules to appropriately behave during inference (for example: nn.Dropout)
    total_samples, correct_predictions = 0, 0
    with torch.no_grad():
        for _, (X,Y) in enumerate(valset_loader):
            X=X.to(device)
            Y=Y.to(device)
            Y_ = net(X)
            loss = loss_fn(Y_, Y)
            __, Y_predicted = Y_.max(1)
            total_samples += Y_predicted.size(0)
            correct_prediction = Y_predicted.eq(Y).sum().item()
            correct_predictions += correct_prediction
    val_acc.append( (correct_predictions/total_samples)*100. )

# training and testing completed. Save the model parameters and plots
torch.save(net.state_dict(),args.output_file)
Save_Stats(train_acc, val_acc, args.normalization)