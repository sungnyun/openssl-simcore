from cProfile import label
import os
import random
import numpy as np

import torch
import torchvision.datasets as datasets
import torch.utils.data as data


class ImageFolderSemiSup(data.Dataset):
    def __init__(self, root='', transform=None, p=1.0, index=None, return_idx=False):
        super(ImageFolderSemiSup, self).__init__()
        
        self.root = root
        self.transform = transform
        self.p = p
        self.return_idx = return_idx
        
        self.train_dataset = datasets.ImageFolder(root=self.root, transform=self.transform)
        if index is None:
            # randomly select samples
            random.seed(0)
            self.dataset_len = int(self.p * len(self.train_dataset))
            self.sampled_index = random.sample(range(len(self.train_dataset)), self.dataset_len)
        else:
            # samples selected by x_u_split function
            self.sampled_index = index
            self.dataset_len = len(self.sampled_index)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        if self.return_idx:
            return index, self.train_dataset[self.sampled_index[index]]
        else:
            return self.train_dataset[self.sampled_index[index]]


def x_u_split(args, include_lb_to_ulb=False):
    """ in TorchSSL, include_lb_to_ulb=True
    """
    labeled_idx = random.sample(range(args.n_data), args.num_labeled)
    if include_lb_to_ulb:
        unlabeled_idx = list(range(args.n_data))
    else:
        unlabeled_idx = list(set(range(args.n_data)) - set(labeled_idx))

    return labeled_idx, unlabeled_idx