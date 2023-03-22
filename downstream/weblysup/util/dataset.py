import os
import random
import numpy as np
from copy import deepcopy

import torch
import torch.utils.data as data
from .transform import TwoCropTransform


class IndexDataset(data.Dataset):
    def __init__(self, train_dataset):
        super(IndexDataset, self).__init__()
        
        self.train_dataset = train_dataset
        self.dataset_len = len(self.train_dataset)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        image, label = self.train_dataset[index]        
        return image, label, index


class MergeDataset(data.Dataset):
    def __init__(self, dataset1, dataset2):
        super(MergeDataset, self).__init__()
        assert isinstance(dataset1, data.Dataset) and isinstance(dataset2, data.Dataset)
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.n1 = len(self.dataset1)
        self.n2 = len(self.dataset2)

    def __len__(self):
        return self.n1 + self.n2

    def __getitem__(self, idx):
        if idx < self.n1:
            return self.dataset1[idx]
        else:
            return self.dataset2[idx - self.n1]


class NoisyMergeDataset(data.Dataset):
    def __init__(self, dataset1, dataset2, probability, index=None, return_prob=False):
        super(NoisyMergeDataset, self).__init__()
        self.dataset1 = deepcopy(dataset1)
        self.dataset2 = deepcopy(dataset2)
        self.n1 = len(self.dataset1)
        self.n2 = len(self.dataset2)
        
        self.dataset1.transform = TwoCropTransform(self.dataset1.transform)
        self.dataset2.transform = TwoCropTransform(self.dataset2.transform)

        self.sampled_index = index
        self.dataset_len = len(self.sampled_index)
        self.probability = probability
        self.return_prob = return_prob

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        idx = self.sampled_index[index]
        if idx < self.n1:
            [image1, image2], label = self.dataset1[idx]
        else:
            [image1, image2], label = self.dataset2[idx - self.n1]
        
        if self.return_prob:
            return image1, image2, label, self.probability[idx]
        else:
            return image1, image2