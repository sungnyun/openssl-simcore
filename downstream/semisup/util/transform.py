import random
import numpy as np
np.random.seed(0)

import torch
import torchvision
from torch import nn
from torchvision.transforms import transforms


# for semi-supervised fine-tuning
class MixMatchTransform:
    def __init__(self, size, mean, std):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip()])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return self.normalize(x1), self.normalize(x2)


class ReMixMatchTransform:
    def __init__(self, size, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip()])
        self.strong = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandAugment(num_ops=2, magnitude=10)])  # TorchSSL used 3, 5
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.normalize(self.weak(x))
        strong_1 = self.normalize(self.strong(x))
        strong_2 = self.normalize(self.strong(x))

        rotate_v_list = [0, 90, 180, 270]
        rotate_v1 = np.random.choice(rotate_v_list, 1).item()
        strong_1_rot = torchvision.transforms.functional.rotate(strong_1, rotate_v1)
        return weak, strong_1, strong_2, strong_1_rot, rotate_v_list.index(rotate_v1)


class FixMatchTransform:
    def __init__(self, size, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
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
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class FlexMatchTransform:
    def __init__(self, size, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip()])
        self.strong = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandAugment(num_ops=2, magnitude=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


_transform_class_map = {
    'mixmatch': MixMatchTransform,
    'remixmatch': ReMixMatchTransform,
    'fixmatch': FixMatchTransform,
    'flexmatch': FlexMatchTransform
}

def get_semisup_transform_class(key):
    if key in _transform_class_map:
        return _transform_class_map[key]
    else:
        raise ValueError('Invalid method: {}'.format(key))
