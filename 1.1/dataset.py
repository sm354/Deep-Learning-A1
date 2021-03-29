import os
import numpy as np
import PIL.Image as Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class Cifar10(Dataset):
    def __init__(self, root='./CIFAR10', train=True, transform=None):
        super(Cifar10, self).__init__()
        self.transform=transform
        path='train' if train else 'val'
        self.data=np.load(root+'%s_x.npy'%path)
        self.targets=np.load(root+'%s_y.npy'%path)
        self.mean=torch.from_numpy(np.load('../CIFAR10/per_pixel_mean.npy')).float()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.astype(np.uint8))
        if self.transform is not None: # Apply transform
            img = self.transform(img)
        img = img - self.mean
        return img, target

    def __len__(self):
        return self.data.shape[0]