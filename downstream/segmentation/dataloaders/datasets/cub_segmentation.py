import os
import numpy as np
import copy
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from dataloaders import custom_transforms as tr


class CUBSegmentationDataset(Dataset):
    def __init__(self, root, test=False, class_label=False):
        ANNOT_PATH = 'segmentations'
        self.image_paths = list()
        self.label_paths = list()
        self.test = test
        self.class_label = class_label
        if self.test:
            split = 'test'
        else:
            split = 'train'
        datapath = os.path.join(root, split)
        for i, cls in enumerate(os.listdir(datapath)):
            for img_name in os.listdir(os.path.join(datapath, cls)):
                self.image_paths.append((os.path.join(datapath, cls, img_name), i+1))
                self.label_paths.append(os.path.join(root, ANNOT_PATH, cls, img_name.replace('.jpg', '.png')))
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path, cls = self.image_paths[idx]
        image = Image.open(path).convert('RGB')
        label = Image.open(self.label_paths[idx])
        sample = {'image': image, 'label': label}

        if not self.test:
            ret = self.transform_tr(sample)
        else:
            ret = self.transform_val(sample)

        new_label = copy.deepcopy(ret['label'])
        new_label[ret['label'] > 204.] = 1.
        new_label[new_label >= 51.] = 255.
        if len(new_label.shape) != 2:
            new_label = new_label[:,:,0]
        if self.class_label:
            new_label[new_label == 1.] *= cls
        ret['label'] = new_label 

        return ret

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=256, crop_size=224),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=224),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

