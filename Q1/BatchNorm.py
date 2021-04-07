import torch
import torch.nn as nn


class BatchNorm2D(nn.Module):
    def __init__(self, num_channels, epsilon = 1e-5, momentum = 0.1, rescale = True):
        super(BatchNorm2D, self).__init__()
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

        # define parameters running mean and variance which is not learnable
        # keep track of mean and variance(but donot learn them), momentum is used which weighs current batch-mean and
        # variance with the running mean and variance using (momentum*runningmean+(1-momentum)*currentmean)
        self.register_buffer('runningmean', torch.zeros(num_channels))
        self.register_buffer('runningvar', torch.ones(num_channels))

    def forward(self, x):
        assert x.shape[1] == self.num_channels
        assert len(x.shape) == 4 #4 because (batchsize, numchannels, height, width)

        if(self.training):
            #calculate mean and variance along the dimensions other than the channel dimension
            #variance calculation is using the biased formula during training
            (variance, mean) = torch.var_mean(x, dim = [0, 2, 3], unbiased = False)
            self.runningmean = (self.momentum) * mean + (1-self.momentum) * self.runningmean 
            self.runningvar = (self.momentum) * variance + (1-self.momentum) * self.runningvar
            out = (x-mean.view([1, self.num_channels, 1, 1]))/torch.sqrt(variance.view([1, self.num_channels, 1, 1])+self.epsilon)

        else:
            out = (x-self.runningmean.view([1, self.num_channels, 1, 1]))/torch.sqrt(self.runningvar.view([1, self.num_channels, 1, 1])+self.epsilon)
            #during testing just use the running mean and variance
        
        if(self.rescale == True):
            out = self.gamma.view([1, self.num_channels, 1, 1]) * out + self.beta.view([1, self.num_channels, 1, 1])
        return out
# m = BatchNorm2D(3)
# m_pytorch = nn.BatchNorm2d(3)
# input = torch.randn(3, 3, 1, 1)
# output = m(input)
# output_pytorch = m_pytorch(input)
# print(output)
# print(output_pytorch)
# print((output - output_pytorch).mean())

