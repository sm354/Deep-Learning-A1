import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from ResNets import *
from Normalisations import *

torch.manual_seed(0) # hopefully this will ensure fair comparison across different norms
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
    np.save(stats_path,data)
    SavePlots(data[0], data[1], 'acc', exp_name)

def SavePlots(y1, y2, metric, exp_name):
    try:
        plt.clf()
    except Exception as e:
        pass
    plt.title(exp_name)
    plt.xlabel('epochs')
    plt.ylabel(metric)
    epochs=np.arange(1,len(y1)+1,1)
    plt.plot(epochs,y1,label='train %s'%metric)
    plt.plot(epochs,y2,label='val %s'%metric)
    ep=np.argmax(y2)
    plt.plot(ep+1,y2[ep],'r*',label='bestacc = %i'%(y2[ep]))
    plt.legend()
    plt.savefig('./ForReport/%s_%s'%(exp_name,metric), dpi=95)

def plot_quantiles(quantiles_data,title):
    try:
        plt.clf()
    except Exception as e:
        pass
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel('quantiles')
    epochs=np.arange(1,quantiles_data.shape[0]+1,1)
    plt.plot(epochs,quantiles_data[:,0],label='1st quantile')
    plt.plot(epochs,quantiles_data[:,1],label='20th quantile')
    plt.plot(epochs,quantiles_data[:,2],label='80th quantile')
    plt.plot(epochs,quantiles_data[:,3],label='99th quantile')
    plt.legend()
    plt.savefig('./ForReport/quantiles/%s_quantiles'%(title), dpi=95)
    np.save('./ForReport/quantiles/%s_quantiles.npy'%(title), quantiles_data)

parser = argparse.ArgumentParser(description='Training ResNets with Normalizations on CIFAR10')
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--normalization', type=str)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--output_file', type=str)
parser.add_argument('--n', help='number of (per) residual blocks', type=int)
parser.add_argument('--r', default=10, help='number of classes', type=int)
parser.add_argument('--quantiles',default=False,help='give argument if quantiles to be plotted')
args=parser.parse_args()

print(args.normalization)

n=args.n
r=args.r
normalization_layers = {'torch_bn':nn.BatchNorm2d,'bn':BatchNorm2D,'in':InstanceNorm2D,'bin':BIN2D,'ln':LayerNorm2D,'gn':GroupNorm2D,'nn':None}
norm_layer_name=args.normalization
norm_layer=normalization_layers[norm_layer_name]

# create the required ResNet model
model = ResNet(n,r,norm_layer_name,norm_layer)
# print(model(torch.rand((2,3,32,32))).shape) : all resnet models working [CHECKED]

device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
model = model.to(device)

# create train-val-test split of CIFAR10
X_train, Y_train, X_val, Y_val, per_pixel_mean = split_cifar(data_dir=args.data_dir[:-19]) # IMPORTANT NOTE - 'cifar-10-batches-py' this should not be in data_dir but submission instructions has it! Either delete string or shout on piazza

def get_transforms(train=True):
    train_transforms= [transforms.ToPILImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),(1.,1.,1.)), ]
    val_transforms= [transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),(1.,1.,1.)), ]
    return train_transforms if train else val_transforms

train_transform = transforms.Compose(get_transforms(train=True)) 
val_transform = transforms.Compose(get_transforms(train=False))

trainset = Cifar10(X_train, Y_train, transform=train_transform)
trainset_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
valset = Cifar10(X_val, Y_val, transform=val_transform)
valset_loader = DataLoader(valset, batch_size=128, shuffle=False, num_workers=2)

start_epoch, end_epoch = 1, args.num_epochs+1
loss_fn = nn.CrossEntropyLoss()
lr=0.1
optimizer = optim.SGD(model.parameters(),lr=lr)
train_acc, val_acc = [], []

if args.quantiles!=False:
    quantiles_data=[] #here 4 quantiles will be added as a tupple i.e. (q_0.01, q_0.2, q_0.8, q_0.99); then finally it will be made a numpy array of shape : num_epochs x 4

for epoch in range(start_epoch, end_epoch):
    if epoch==75 or epoch==95:
        lr/=10.
        optimizer=optim.SGD(model.parameters(),lr=lr)
    # Training 
    model.train()
    total_samples, correct_predictions = 0, 0
    for _, (X,Y) in enumerate(trainset_loader):
        X=X.to(device)
        Y=Y.to(device)
        optimizer.zero_grad() # remove history
        Y_ = model(X)
        Y_predicted = Y_.argmax(dim=1)
        
        loss = loss_fn(Y_, Y)
        loss.backward() # create computational graph i.e. find gradients
        optimizer.step() # update weights/biases
        
        # __, Y_predicted = Y_.max(1)
        correct_prediction = Y_predicted.eq(Y).sum()
        correct_predictions += correct_prediction.item()
        total_samples += Y_predicted.size(0)
    train_acc.append( (correct_predictions/total_samples)*100. )

    # Testing
    model.eval() #this is useful in informing nn.modules to appropriately behave during inference (for example: nn.Dropout)
    total_samples, correct_predictions = 0, 0
    with torch.no_grad():
        if args.quantiles!=False:
            quantiles_dataset=[] #here 64*10000 samples will be concatenated
        for _, (X,Y) in enumerate(valset_loader):
            X=X.to(device)
            Y=Y.to(device)
            
            # will be used for plotting of quantiles [controllable using argument parser]
            if args.quantiles==False:
                Y_ = model(X)
            else:
                Y_,max_pool_output = model(X, quantiles=True)
                quantiles_dataset.append(max_pool_output)

            Y_predicted = Y_.argmax(dim=1)
            loss = loss_fn(Y_, Y)

            # __, Y_predicted = Y_.max(1)
            correct_prediction = Y_predicted.eq(Y).sum()
            correct_predictions += correct_prediction.item()
            total_samples += Y_predicted.size(0)
        if args.quantiles!=False:
            # find the required quantiles
            quantiles_dataset=torch.cat(quantiles_dataset).cpu()
            assert quantiles_dataset.shape[0] == 640000
            quantiles_data.append((np.quantile(quantiles_dataset,0.01),np.quantile(quantiles_dataset,0.2),np.quantile(quantiles_dataset,0.8),np.quantile(quantiles_dataset,0.99)))

    val_acc.append( (correct_predictions/total_samples)*100. )
    print("epoch ", epoch, "train acc", train_acc[-1], "val acc", val_acc[-1])

# training and testing completed. Save the model parameters and plots
torch.save(model.state_dict(),args.output_file)
Save_Stats(train_acc, val_acc, args.normalization)
if args.quantiles != False:
    quantiles_data=np.array(quantiles_data) #num_epochs x 4
    plot_quantiles(quantiles_data,title=args.normalization)
