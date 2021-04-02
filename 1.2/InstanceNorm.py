import torch
import torch.nn as nn


class InstanceNorm2D(nn.Module):
    def __init__(self, num_channels, epsilon = 1e-5, momentum = 0.1, rescale = True):
        super(InstanceNorm2D, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.momentum = momentum
        self.rescale = rescale

        if(self.rescale == True):
            # define parameters gamma, beta which are learnable        
            # dimension of gamma and beta should be (num_channels) ie its a one dimensional vector
            # initializing gamma as ones vector and beta as zeros vector (implies no scaling/shifting at the start)
            self.gamma = nn.Parameter(torch.ones(num_channels))
            self.beta = nn.Parameter(torch.zeros(num_channels))
        
        # running mean and variance should have the same dimension as in batchnorm
        # ie, a vector of size num_channels because while testing, when we get one
        # sample at a time, we should be able to use this.
        self.register_buffer('runningmean', torch.zeros(num_channels))
        self.register_buffer('runningvar', torch.ones(num_channels))

    def forward(self, x):
        assert x.shape[1] == self.num_channels
        assert len(x.shape) == 4 # 4 because len((batchsize, numchannels, height, width)) = 4

        if(self.training):
            #calculate mean and variance along the dimensions other than the channel dimension
            #variance calculation is using the biased formula during training
            (variance, mean) = torch.var_mean(x, dim = [2, 3], unbiased=False)
            out = (x-mean.view([-1, self.num_channels, 1, 1]))/torch.sqrt(variance.view([-1, self.num_channels, 1, 1])+self.epsilon)

            #logically the variance and mean calculation for updating running variance, mean should be as follows
            (variance, mean) = torch.var_mean(x, dim = [0, 2, 3], unbiased = False)
            self.runningmean = (self.momentum) * mean + (1-self.momentum) * self.runningmean 
            self.runningvar = (self.momentum) * variance + (1-self.momentum) * self.runningvar

        else:
            out = (x-self.runningmean.view([1, self.num_channels, 1, 1]))/torch.sqrt(self.runningvar.view([11, self.num_channels, 1, 1])+self.epsilon)
        
        if(self.rescale == True):
            out = self.gamma.view([1, self.num_channels, 1, 1]) * out + self.beta.view([1, self.num_channels, 1, 1])
            return out
        else:
            return out
        #during testing just use the running mean and variance


m = InstanceNorm2D(3, rescale = True)
m_pytorch = nn.InstanceNorm2d(3, affine = True)
for i in range(10):
    input = torch.randn(3, 3, i, i)
    output_pytorch = m_pytorch(input)
    # print(output_pytorch)
    output = m(input)
    # print(output)
    print((output - output_pytorch).mean())