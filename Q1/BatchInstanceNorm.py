import torch
import torch.nn as nn
from BatchNorm import BatchNorm2D
from InstanceNorm import InstanceNorm2D

class BIN2D(nn.Module):
    def __init__(self, num_channels, epsilon = 1e-5, momentum = 0.1,):
        super(BIN2D, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.momentum = momentum

        self.batchnorm = BatchNorm2D(self.num_channels, epsilon = self.epsilon, momentum = self.momentum, rescale= False)
        self.instancenorm = InstanceNorm2D(self.num_channels, epsilon = self.epsilon, momentum = self.momentum, rescale= False)

        # the gate variable to be learnt
        self.rho = nn.Parameter(torch.ones(self.num_channels))

        #gamma and beta for rescaling
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        # clip all elements of rho between 0 and 1

        self.rho = nn.Parameter(torch.clamp(self.rho, 0, 1))
        xbn = self.batchnorm(x)
        xin = self.instancenorm(x)
        out = self.rho.view([1, self.num_channels, 1, 1]) * xbn + self.rho.view([1, self.num_channels, 1, 1]) * xin
        out = self.gamma.view([1, self.num_channels, 1, 1]) * out + self.beta.view([1, self.num_channels, 1, 1])

        return out





