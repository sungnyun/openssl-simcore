"""
Cited from: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms as tf
from glob import glob


class ImageNetSubset(data.Dataset):
    def __init__(self, subset_file, root='', split='train', 
                    transform=None):
        super(ImageNetSubset, self).__init__()

        self.root = root
        self.transform = transform
        self.split = split

        # Read the subset of classes to include (sorted)
        with open(subset_file, 'r') as f:
            result = f.read().splitlines()
        subdirs = []
        for line in result:
            subdirs.append(line)

        # Gather the files (sorted)
        imgs = []
        for i, subdir in enumerate(subdirs):
            subdir_path = os.path.join(self.root, subdir)
            files = sorted(glob(os.path.join(self.root, subdir, '*.JPEG')))
            for f in files:
                imgs.append((f, i))
        self.imgs = imgs
    
	# Resize
        self.resize = tf.Resize(256)

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img) 
        return img

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        im_size = img.size
        img = self.resize(img) 

        if self.transform is not None:
            img = self.transform(img)

        return img, target
