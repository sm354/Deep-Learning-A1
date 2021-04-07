import torch
import torch.nn as nn


class LayerNorm2D(nn.Module):
    def __init__(self, end_dimension, epsilon = 1e-5, momentum = 0.1):
        super(LayerNorm2D, self).__init__()
        self.end_dimension = end_dimension
        self.epsilon = epsilon
        self.momentum = momentum

        self.gamma = nn.Parameter(torch.ones(end_dimension))
        self.beta = nn.Parameter(torch.zeros(end_dimension))

    def forward(self, x):
        assert list(x.shape)[1:] == self.end_dimension
        assert len(x.shape) == 4 # 4 because len((batchsize, numchannels, height, width)) = 4

        (variance, mean) = torch.var_mean(x, dim = [1, 2, 3], unbiased=False)
        out = (x-mean.view([-1, 1, 1, 1]))/torch.sqrt(variance.view([-1, 1, 1, 1])+self.epsilon)

        out = self.gamma.unsqueeze(0) * out + self.beta.unsqueeze(0)
        return out