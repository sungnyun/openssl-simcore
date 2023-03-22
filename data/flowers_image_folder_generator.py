import glob
import re
import os
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import shutil
import copy
import scipy.io


ROOT = "./data"
metadata = scipy.io.loadmat(os.path.join(ROOT, "imagelabels.mat"))
setid = scipy.io.loadmat(os.path.join(ROOT, "setid.mat"))

metadata = metadata['labels'][0]
train_setid = setid['trnid'][0]
valid_setid = setid['valid'][0]
test_setid = setid['tstid'][0]


train_paths_by_class = defaultdict(list)
valid_paths_by_class = defaultdict(list)
test_paths_by_class = defaultdict(list)
for idx, label in tqdm(enumerate(metadata, start=1)):
    path = os.path.join(ROOT, 'flowers', 'image_{:05d}.jpg'.format(idx))
    if idx in train_setid:
        train_paths_by_class[str(label)].append(path)
    elif idx in valid_setid:
        valid_paths_by_class[str(label)].append(path)
    elif idx in test_setid:
        test_paths_by_class[str(label)].append(path)
    else:
        raise ValueError(idx)

TARGET_DIR = './train'
print("Copying images...")
for cls, paths in train_paths_by_class.items():
    for source_path in paths:
        path = source_path.split('/')[-1]
        target_path = os.path.join(TARGET_DIR, cls, path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy(source_path, target_path)
print("ImageFolder directory created at {}".format(os.path.abspath(TARGET_DIR)))

TARGET_DIR = './val'
print("Copying images...")
for cls, paths in valid_paths_by_class.items():
    for source_path in paths:
        path = source_path.split('/')[-1]
        target_path = os.path.join(TARGET_DIR, cls, path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy(source_path, target_path)
print("ImageFolder directory created at {}".format(os.path.abspath(TARGET_DIR)))

TARGET_DIR = "./test"
print("Copying images...")
for cls, paths in test_paths_by_class.items():
    for source_path in paths:
        path = source_path.split('/')[-1]
        target_path = os.path.join(TARGET_DIR, cls, path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy(source_path, target_path)
print("ImageFolder directory created at {}".format(os.path.abspath(TARGET_DIR)))

### Option: use train + val instead of only train

SOURCE_DIR = './flowers/val'
TARGET_DIR = './flowers/train'

for dirlist in os.listdir(SOURCE_DIR):
    for file in os.listdir(os.path.join(SOURCE_DIR, dirlist)):
        shutil.move(os.path.join(SOURCE_DIR, dirlist, file), os.path.join(TARGET_DIR, dirlist, file))
