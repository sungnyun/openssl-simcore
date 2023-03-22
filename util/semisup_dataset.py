from cProfile import label
import os
import random
import numpy as np

import torch
import torchvision.datasets as datasets
import torch.utils.data as data


class ImageFolderSemiSup(data.Dataset):
    def __init__(self, root='', transform=None, p=1.0):
        super(ImageFolderSemiSup, self).__init__()
        
        self.root = root
        self.transform = transform
        self.p = p
        
        self.train_dataset = datasets.ImageFolder(root=self.root, transform=self.transform)
        
        # randomly select samples
        random.seed(0)
        self.dataset_len = int(self.p * len(self.train_dataset))
        self.sampled_index = random.sample(range(len(self.train_dataset)), self.dataset_len)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        return self.train_dataset[self.sampled_index[index]]