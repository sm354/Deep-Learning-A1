import torch
import torch.nn as nn
import torch.nn.functional as F

# implementation details taken from paper:
# 1. We adopt batch normalization (BN) right after each convolution and before activation
# 2. In residual block : We adopt the second nonlinearity after the addition

# NOTE : the biases are included in every conv layer (nothing was said in paper about biases but online resources don't add it)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm_layer_name, norm_layer):
        super(ResidualBlock, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.stride=stride
        self.shortcut=nn.Sequential()

        # a residual block comprises of 2 conv layers 
        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=1) 
        self.bn1=nn.Identity() if norm_layer==None else norm_layer(out_channels)
        self.conv2=nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1) 
        self.bn2=nn.Identity() if norm_layer==None else norm_layer(out_channels)
        if in_channels != out_channels: 
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride),
                nn.Identity() if norm_layer==None else norm_layer(out_channels)
            )

    def forward(self, inp):
        out=self.conv1(inp)
        out=self.bn1(out)
        out=F.relu(out)
        out=self.bn2(self.conv2(out)) 
        if self.in_channels != self.out_channels: 
            out = out + self.shortcut(inp)
        else:
            out = out + inp # dimensions match 
        out = F.relu(out) # second nonlinearity in the residual block after the addition
        return out


class ResNet(nn.Module):
    def __init__(self, n, r, norm_layer_name, norm_layer): 
        # norm_layer_name is a string, norm_layer is nn.Module
        # norm_layer_name can be used in hyperparam tuning of G in GN normalisation
        super(ResNet, self).__init__()
        self.n=n
        self.r=r
        self.conv1=nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1) #fist conv layer
        self.bn1=nn.Identity() if norm_layer==None else norm_layer(16)
        
        #6n layers : each 2n layers of same feature map size
        blocks_A, blocks_B, blocks_C = [ResidualBlock(16,16,1,norm_layer_name,norm_layer)],[ResidualBlock(16,32,2,norm_layer_name, norm_layer)],[ResidualBlock(32,64,2,norm_layer_name, norm_layer)]
        for i in range(1,n):
            blocks_A.append(ResidualBlock(16,16,1,norm_layer_name, norm_layer))
            blocks_B.append(ResidualBlock(32,32,1,norm_layer_name, norm_layer))
            blocks_C.append(ResidualBlock(64,64,1,norm_layer_name, norm_layer))
        
        self.blocks_A = nn.Sequential(*blocks_A) #feature map size 32,32
        self.blocks_B = nn.Sequential(*blocks_B) #feature map size 16,16
        self.blocks_C = nn.Sequential(*blocks_C) #feature map size 8,8

        self.GAP = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(in_features=64,out_features=r)

    def forward(self, inp, quantiles=False):
        # if quantiles argument is passed (which will be done during model validation during gradient descent)
        # then return the flattened (along batch as well) vector output of max_pool layer
        out=self.conv1(inp)
        out=self.bn1(out)
        out=F.relu(out)
        out=self.blocks_A(out)
        out=self.blocks_B(out)
        out=self.blocks_C(out)
        out=self.GAP(out)
        out=out.view(out.size(0),-1)
        max_pool_output=out.view(-1)
        out=self.fc(out)
        if quantiles==False:
            return out
        else:
            return (out, max_pool_output)