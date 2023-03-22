import random
import numpy as np
np.random.seed(0)

import torch
import torchvision
from torch import nn
from torchvision.transforms import transforms


class SelfTrainingTransform:
    def __init__(self, size,  mean, std):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        return self.transform(x)


class OpenMatchTransform:
    def __init__(self, size, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip()])

        self.weak2 = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip()])

        self.strong = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandAugment(num_ops=2, magnitude=10)])
            # RandAugmentMC(n=2, m=10)])  # refer to original code in https://github.com/kekmodel/FixMatch-pytorch/blob/master/dataset/randaugment.py

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        weak2 = self.weak2(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong), self.normalize(weak2)


_transform_class_map = {
    'self_training': SelfTrainingTransform,
    'openmatch': OpenMatchTransform,
}

def get_semisup_transform_class(key):
    if key in _transform_class_map:
        return _transform_class_map[key]
    else:
        raise ValueError('Invalid method: {}'.format(key))
