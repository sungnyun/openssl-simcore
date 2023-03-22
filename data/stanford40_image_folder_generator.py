from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import os
import shutil
from shutil import copyfile

# Change the path to your dataset folder:
base_folder = './JPEGImages/'

train_path = './ImageSplits/train.txt'
test_path = './ImageSplits/test.txt'

# Here declare where you want to place the train/test folders
# You don't need to create them!
test_folder = './data/stanford40/test/'
train_folder = './data/stanford40/train/'

os.makedirs(test_folder, exist_ok=True)
os.makedirs(train_folder, exist_ok=True)

with open(train_path) as f:
    train_lines = f.readlines()

with open(test_path) as f:
    test_lines = f.readlines()

test_images, train_images = 0,0

for train_line in train_lines:
    file_name = train_line.strip()
    class_name = '_'.join(train_line.strip().split('_')[:-1])
    if os.path.exists(base_folder+file_name):
        os.makedirs(train_folder+class_name, exist_ok=True) 
        copyfile(base_folder+file_name, os.path.join(train_folder,class_name,file_name))
        train_images += 1

for test_line in test_lines:
    file_name = test_line.strip() 
    class_name = '_'.join(test_line.strip().split('_')[:-1])
    if os.path.exists(base_folder+file_name):
        os.makedirs(test_folder+class_name, exist_ok=True) 
        copyfile(base_folder+file_name, os.path.join(test_folder,class_name,file_name))
        test_images += 1

print(train_images, test_images)
assert train_images == 4000
assert test_images == 5532

print('Dataset succesfully splitted!')
