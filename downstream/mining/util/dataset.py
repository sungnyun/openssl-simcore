import torch
import torch.utils.data as data


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