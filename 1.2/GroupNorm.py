import torch
import torch.nn as nn


class GroupNorm2D(nn.Module):
    def __init__(self, num_groups, num_channels, epsilon = 1e-5):
        super(GroupNorm2D, self).__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.epsilon = epsilon

        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        assert x.shape[1] == self.num_channels
        assert len(x.shape) == 4 #4 because (batchsize, numchannels, height, width)
        
        [N, C, H, W] = list(x.shape)
        out = torch.reshape(x, (N, self.num_groups, self.num_channels//self.num_groups, H, W))
        (variance, mean) = torch.var_mean(out, dim = [2, 3, 4], unbiased=False, keepdim=True)
        out = (out-mean)/torch.sqrt(variance +self.epsilon)
        out = out.view(N, self.num_channels, H, W)
        out = self.gamma.view([1, self.num_channels, 1, 1]) * out + self.beta.view([1, self.num_channels, 1, 1])

        return out