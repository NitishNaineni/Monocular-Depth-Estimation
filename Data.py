from itertools import permutations
import mat73
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
from PIL import Image
from torchvision import transforms
import torch


class Depthset(Dataset):
    def __init__(self,data_dict,transform=None):
        self.data_dict = data_dict
        self.size = self.data_dict['images'].shape[-1]
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self,idx):
        image = self.data_dict["images"][:,:,:,idx]
        depth = self.data_dict["depths"][:,:,idx]
        sample = {"image":image,"depth":depth}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Agument(object):
    def __init__(self,probability):
        self.perms = list(permutations(range(3),3))
        self.prob = probability

    def __call__(self,sample):
        image,depth = sample['image'], sample['depth']
        
        alt_method = random.randint(0,2)

        image = {
            0:image,
            1:np.fliplr(image),
            2:np.flipud(image)
        }[alt_method]
        
        depth = {
            0:depth,
            1:np.fliplr(depth),
            2:np.flipud(depth)
        }[alt_method]

        if random.random() < self.prob:
            image = image[...,list(self.perms[random.randint(0, len(self.perms) -1)])]

        sample = {"image":image,"depth":depth}

        return sample

class toTensor(object):
    def __init__(self,downscale=2):
        self.downscale = downscale

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = self.to_tensor(image)

        depth = depth[::self.downscale, ::self.downscale]
        depth = self.to_tensor(depth) * 1000

        depth = torch.clamp(depth,10,1000)
        sample = {'image': image, 'depth': depth}

        return sample

    def to_tensor(self,img):
        if img.ndim == 2:
            img = img[...,np.newaxis]

        img = torch.from_numpy(np.ascontiguousarray(img.transpose((2, 0, 1))))
        return img.float().div(255)
