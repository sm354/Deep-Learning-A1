import torch
import torch.nn as nn

class BatchNorm2D(nn.Module):
    def __init__(self, num_channels, epsilon = 1e-5, momentum = 0.9, rescale = True):
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
            variance = torch.var(x, dim = [0, 2, 3], unbiased=False)
            mean  = torch.mean(x, dim = [0, 2, 3])
            self.runningmean = (1 - self.momentum) * mean + (self.momentum) * self.runningmean 
            self.runningvar = (1 - self.momentum) * variance + (self.momentum) * self.runningvar
            out = (x-mean.view([1, self.num_channels, 1, 1]))/torch.sqrt(variance.view([1, self.num_channels, 1, 1])+self.epsilon)

        else:
            m = x.shape[0]*x.shape[2]*x.shape[3]
            out = (x-self.runningmean.view([1, self.num_channels, 1, 1]))/torch.sqrt((m/(m-1))*self.runningvar.view([1, self.num_channels, 1, 1])+self.epsilon)
            #during testing just use the running mean and (UnBiased) variance 
        
        if(self.rescale == True):
            out = self.gamma.view([1, self.num_channels, 1, 1]) * out + self.beta.view([1, self.num_channels, 1, 1])
        return out

class InstanceNorm2D(nn.Module):
    def __init__(self, num_channels, epsilon = 1e-5, momentum = 0.9, rescale = True):
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
            variance, mean = torch.var(x, dim = [2, 3], unbiased=False), torch.mean(x, dim = [2, 3])
            out = (x-mean.view([-1, self.num_channels, 1, 1]))/torch.sqrt(variance.view([-1, self.num_channels, 1, 1])+self.epsilon)
            
            # SHUBHAM : not needed ---
            # #logically the variance and mean calculation  for updating running variance, mean should be as follows
            # (variance, mean) = torch.var_mean(x, dim = [0, 2, 3], unbiased = False)
            # self.runningmean = (1-self.momentum) * mean + (self.momentum) * self.runningmean 
            # self.runningvar = (1-self.momentum) * variance + (self.momentum) * self.runningvar
            # SHUBHAM : not needed ---

        else:
            variance, mean = torch.var(x, dim = [2, 3], unbiased=False), torch.mean(x, dim = [2, 3])
            out = (x-mean.view([-1, self.num_channels, 1, 1]))/torch.sqrt(variance.view([-1, self.num_channels, 1, 1])+self.epsilon)
            
            # SHUBHAM : not needed ---
            # out = (x-self.runningmean.view([1, self.num_channels, 1, 1]))/torch.sqrt(self.runningvar.view([11, self.num_channels, 1, 1])+self.epsilon)
            # SHUBHAM : not needed ---
        
        if(self.rescale == True):
            out = self.gamma.view([1, self.num_channels, 1, 1]) * out + self.beta.view([1, self.num_channels, 1, 1])
        return out


class BIN2D(nn.Module):
    def __init__(self, num_channels, epsilon = 1e-5, momentum = 0.9):
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
        out = self.rho.view([1, self.num_channels, 1, 1]) * xbn + (1 - self.rho.view([1, self.num_channels, 1, 1]) ) * xin
        out = self.gamma.view([1, self.num_channels, 1, 1]) * out + self.beta.view([1, self.num_channels, 1, 1])
        return out

class LayerNorm2D(nn.Module):
    def __init__(self, num_channels, epsilon = 1e-5):
        super(LayerNorm2D, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon

        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        assert list(x.shape)[1] == self.num_channels
        assert len(x.shape) == 4 # 4 because len((batchsize, numchannels, height, width)) = 4

        variance, mean = torch.var(x, dim = [1,2, 3], unbiased=False), torch.mean(x, dim = [1,2, 3])
        out = (x-mean.view([-1, 1, 1, 1]))/torch.sqrt(variance.view([-1, 1, 1, 1])+self.epsilon)

        out = self.gamma.view([1, self.num_channels, 1, 1]) * out + self.beta.view([1, self.num_channels, 1, 1])
        return out

class GroupNorm2D(nn.Module):
    def __init__(self, num_channels, num_groups=4, epsilon=1e-5):
        super(GroupNorm2D, self).__init__()
        self.num_channels = num_channels
        # self.num_groups = num_groups
        self.num_groups = num_channels // 4
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        assert x.shape[1] == self.num_channels
        assert len(x.shape) == 4 #4 because (batchsize, numchannels, height, width)
        
        [N, C, H, W] = list(x.shape)
        out = torch.reshape(x, (N, self.num_groups, self.num_channels//self.num_groups, H, W))
        variance, mean = torch.var(out, dim = [2, 3, 4], unbiased=False, keepdim=True), torch.mean(out, dim = [2, 3, 4], keepdim=True)
        out = (out-mean)/torch.sqrt(variance +self.epsilon)
        out = out.view(N, self.num_channels, H, W)
        out = self.gamma.view([1, self.num_channels, 1, 1]) * out + self.beta.view([1, self.num_channels, 1, 1])
        return out