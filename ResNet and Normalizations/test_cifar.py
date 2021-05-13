import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import numpy as np
import pandas as pd
import argparse
from ResNets import *
from Normalisations import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
torch.manual_seed(0) # hopefully this will ensure fair comparison across different norms

def print_final_metrics(model, device, X_complete, Y_complete, dataset_name):
	XY = TensorDataset(X_complete, Y_complete)
	Ypred = []
	loader = DataLoader(XY, batch_size=128)

	model.eval() #this is useful in informing nn.modules to appropriately behave during inference (for example: nn.Dropout)
	with torch.no_grad():
		for _, (X,Y) in enumerate(loader):
			X=X.to(device)
			Y=Y.to(device)
			Y_=model(X)
			Y_predicted = Y_.argmax(dim=1)
			Ypred.append(Y_predicted)
	
	Ypred=torch.cat(Ypred)
	assert Ypred.shape[0]==Y_complete.shape[0]
	acc=accuracy_score(Y_complete.cpu().numpy(),Ypred.cpu().numpy())
	micro_f1=f1_score(Y_complete.cpu().numpy(),Ypred.cpu().numpy(),average='micro')
	macro_f1=f1_score(Y_complete.cpu().numpy(),Ypred.cpu().numpy(),average='macro')
	print(dataset_name, "accuracy: ", acc,"micro f1: ", micro_f1, "macro f1: ", macro_f1)

def compute_final_metrics(model, device, per_pixel_mean, data_dir):
	t = transforms.Compose([transforms.ToTensor(),])
	ts = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=t)
	loader = DataLoader(ts, batch_size=40000, shuffle=True, num_workers=2)
	Xs,Ys=[],[] #contains train and val set (torch.tensor)
	for _, (X,Y) in enumerate(loader):
		X-=per_pixel_mean
		Xs.append(X)
		Ys.append(Y)
	
	ts = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=t)
	loader = DataLoader(ts, batch_size=10000, shuffle=True, num_workers=2)
	Xtest,Ytest = next(iter(loader))
	Xtest -= per_pixel_mean

	# now we have Xtrain,Ytrain Xval,Yval Xtest,Ytest
	# compute metrics on them
	print_final_metrics(model, device, Xs[0], Ys[0], 'train set')
	print_final_metrics(model, device, Xs[1], Ys[1], 'validation set')
	print_final_metrics(model, device, Xtest, Ytest, 'test set')


parser = argparse.ArgumentParser(description='Testing ResNets with Normalizations on CIFAR10')
parser.add_argument('--normalization', type=str)
parser.add_argument('--model_file', type=str)
parser.add_argument('--test_data_file', type=str)
parser.add_argument('--output_file', type=str)
parser.add_argument('--n', help='number of (per) residual blocks', type=int)
parser.add_argument('--r', default=10, help='number of classes', type=int)
parser.add_argument('--fm_dir', default=False, help='pass data_dir if final metrics over train-val-test dataset is to be computed')
args=parser.parse_args()

n=args.n
r=args.r
normalization_layers = {'torch_bn':nn.BatchNorm2d,'bn':BatchNorm2D,'in':InstanceNorm2D,'bin':BIN2D,'ln':LayerNorm2D,'gn':GroupNorm2D,'nn':None}
norm_layer_name=args.normalization
norm_layer=normalization_layers[norm_layer_name]

model = ResNet(n,r,norm_layer_name,norm_layer)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
per_pixel_mean = torch.tensor((0.4914, 0.4822, 0.4465)).view([1,3,1,1])

if os.path.exists(args.model_file):        
	model.load_state_dict(torch.load(args.model_file,map_location=device))
print(args.normalization,"model loaded")

# if final metrics are to be computed, then data directory has to be passed :-
if args.fm_dir != False:
	compute_final_metrics(model, device, per_pixel_mean, args.fm_dir)


print("testing the model on unseen data (loaded from csv file)")
X_test = torch.from_numpy(pd.read_csv(args.test_data_file,header=None).values.reshape((-1,3,32,32))/255.).float() - per_pixel_mean
X_complete = TensorDataset(X_test)
Ypred = []
loader = DataLoader(X_complete, batch_size=64)
model.eval() #this is useful in informing nn.modules to appropriately behave during inference (for example: nn.Dropout)
with torch.no_grad():
	for _, X in enumerate(loader): # X is a list of size 1, having tensor of shape: batch_size x 3 x 32 x 32
		X=X[0].to(device) 
		Y_=model(X)
		Y_predicted = Y_.argmax(dim=1)
		Ypred.append(Y_predicted)

Ypred=torch.cat(Ypred).cpu().numpy().reshape((-1,1))
np.savetxt(args.output_file, Ypred, fmt='%d')



