import torch


class TwoCropTransform:
    """ Create two crops of the same image """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]