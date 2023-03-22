import random
import numpy as np
np.random.seed(0)

import torch
import torchvision
from torch import nn
from torchvision.transforms import transforms


# for self-supervised pretraining
class TwoCropTransform:
    """ Create two crops of the same image """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class MultiCropTransform:
    """ Create multi crops of the same image """
    def __init__(self, local_crops_number):
        # slightly modified transforms (e.g., Crop scale, ColorJitter, Blur, Solarize)
        global_crops_scale = (0.2, 1.)
        local_crops_scale = (0.05, 0.2)
        self.local_crops_number = local_crops_number

        self.global_transform = transforms.Compose([
                                    transforms.RandomResizedCrop(size=224, scale=global_crops_scale),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                                    transforms.RandomGrayscale(p=0.2),
                                    # transforms.GaussianBlur(kernel_size=(3,3)), # require too much cost
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                ])
        self.local_transform = transforms.Compose([
                                    transforms.RandomResizedCrop(size=96, scale=local_crops_scale),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                                    transforms.RandomGrayscale(p=0.2),
                                    # transforms.GaussianBlur(kernel_size=(3,3)), # require too much cost
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                ])

    def __call__(self, x):
        crops = []
        crops.append(self.global_transform(x))
        crops.append(self.global_transform(x))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(x))
        return crops


class MAETransform:
    """ Single-view transform for MAE-ViT """
    def __init__(self, img_size):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.transform(x)