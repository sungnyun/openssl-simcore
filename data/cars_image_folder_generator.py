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

ROOT = "./cars"

LABEL = "type" # brand, type

metadata = scipy.io.loadmat(os.path.join(ROOT, "raw/cars_annos.mat"))
type_list = ['SUV', 'Sedan', 'Hatchback', 'Convertible', 'Coupe', 'Wagon', 'Cab', 'Van', 'Minivan']
UNK = {'Type-S': 'Sedan', 'R': 'Coupe', 'GS': 'Sedan', 'ZR1': '', 'Z06': '', 'SS': '', 'SRT-8': '', 'SRT8': '', 'Abarth': '', 'SuperCab': 'Cab',
       'IPL': 'Coupe', 'XKR': 'Coupe', 'Superleggera': 'Coupe'}

if LABEL in ['brand', 'type']:
    if LABEL == 'brand':
        cls_to_label = {i+1: cls[0].split(' ')[0] for i, cls in enumerate(metadata["class_names"][0])}
    elif LABEL == 'type':
        cls_to_label = {}
        for i, cls in enumerate(metadata["class_names"][0]):
            cartype = cls[0].split(' ')[-2]
            if cartype not in type_list:
                if UNK[cartype]:
                    cartype = UNK[cartype]
            cls_to_label[i+1] = cartype
            # = {i+1: cls[0].split(' ')[-2] for i, cls in enumerate(metadata["class_names"][0])}

metadata = metadata["annotations"][0]

train_paths_by_class = defaultdict(list)
test_paths_by_class = defaultdict(list)

for m in tqdm(metadata):
    path, _, _, _, _, cls, is_test = m
    cls = cls.item()
    path = path.item()
    is_test = is_test.item()
    
    if LABEL == 'original':
        key = str(cls)
    else:
        key = cls_to_label[cls]
        if LABEL == 'type' and key not in type_list:
            # print(key)
            continue

    if is_test:
        test_paths_by_class[key].append(path)
    else:
        train_paths_by_class[key].append(path)

i = 0
for key in train_paths_by_class:
    i += len(train_paths_by_class[key])
print(i)

# SOURCE_DIR = os.path.join(ROOT, "car_ims")
# TARGET_DIR = "./train"
SOURCE_DIR = os.path.join(ROOT, "raw/car_ims")
TARGET_DIR = os.path.join(ROOT, "./{}/train".format(LABEL))

print("Copying images...")
for cls, paths in train_paths_by_class.items():
    for source_path in paths:
        path = source_path.split('/')[1]
        target_path = os.path.join(TARGET_DIR, cls, path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy(os.path.join(SOURCE_DIR, path), target_path)
print("ImageFolder directory created at {}".format(os.path.abspath(TARGET_DIR)))


# SOURCE_DIR = os.path.join(ROOT, "car_ims")
# TARGET_DIR = "./test"
SOURCE_DIR = os.path.join(ROOT, "raw/car_ims")
TARGET_DIR = os.path.join(ROOT, "./{}/test".format(LABEL))

print("Copying images...")
for cls, paths in test_paths_by_class.items():
    for source_path in paths:
        path = source_path.split('/')[1]
        target_path = os.path.join(TARGET_DIR, cls, path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy(os.path.join(SOURCE_DIR, path), target_path)
print("ImageFolder directory created at {}".format(os.path.abspath(TARGET_DIR)))
