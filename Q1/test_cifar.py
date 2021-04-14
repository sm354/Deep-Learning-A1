import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import PIL.Image as Image
import numpy as np
import pandas as pd
import argparse
from ResNets import *
from Normalisations import *


parser = argparse.ArgumentParser(description='Testing ResNets with Normalizations on CIFAR10')
parser.add_argument('--normalization', type=str)
parser.add_argument('--model_file', type=str)
parser.add_argument('--test_data_file', type=str)
parser.add_argument('--output_file', type=str)
parser.add_argument('--n', help='number of (per) residual blocks', type=int)
parser.add_argument('--r', default=10, help='number of classes', type=int)
args=parser.parse_args()

n=args.n
r=args.r
normalization_layers = {'torch_bn':nn.BatchNorm2d,'bn':BatchNorm2D,'in':InstanceNorm2D,'bin':BIN2D,'ln':LayerNorm2D,'gn':GroupNorm2D,'nn':None}
norm_layer_name=args.normalization
norm_layer=normalization_layers[norm_layer_name]

net = ResNet(n,r,norm_layer_name,norm_layer)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if os.path.exists(args.model_file):        
	net.load_state_dict(torch.load(args.model_file))

X_test = torch.from_numpy(pd.read_csv(args.test_data_file).values.reshape((-1,3,32,32))/255.).float()
X_test = X_test.to(device)

_,Y_pred = net(X_test).max(1)
Y_pred = Y_pred.cpu().numpy().reshape((-1,1))

df=pd.DataFrame(Y_pred)
df.to_csv(args.output_file,index=False,header=False)



