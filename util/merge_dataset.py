import torch.utils.data as data
from torchvision import datasets


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


class MergeAllDataset(data.Dataset):
    def __init__(self, transform):
        super(MergeAllDataset, self).__init__()
        self.transform = transform

        print('Merging COCO, iNaturalist, ImageNet, and Places365...')
        self.dataset1 = datasets.ImageFolder(root='/path/to/coco/train',
                                             transform=self.transform)
        self.dataset2 = datasets.INaturalist(root='/path/to/iNaturalist',
                                             version='2021_train_mini',
                                             transform=self.transform)
        self.dataset3 = datasets.ImageFolder(root='/path/to/ILSVRC2015/ILSVRC2015/Data/CLS-LOC/',
                                             transform=self.transform)
        self.dataset4 = datasets.ImageFolder(root='/path/to/places/data_256/train',
                                             transform=self.transform)

        self.n1 = len(self.dataset1)
        self.n2 = len(self.dataset2) + self.n1
        self.n3 = len(self.dataset3) + self.n2
        self.n4 = len(self.dataset4) + self.n3

    def __len__(self):
        return self.n4

    def __getitem__(self, idx):
        if idx < self.n1:
            return self.dataset1[idx]
        elif idx >= self.n1 and idx < self.n2:
            return self.dataset2[idx - self.n1]
        elif idx >= self.n2 and idx < self.n3:
            return self.dataset3[idx - self.n2]
        else:
            return self.dataset4[idx - self.n3]
