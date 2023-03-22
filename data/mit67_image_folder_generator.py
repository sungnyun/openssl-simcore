from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import os
import shutil
from shutil import copyfile

# Change the path to your dataset folder:
base_folder = './Images/'

train_path = './TrainImages.txt'
test_path = './TestImages.txt'

# Here declare where you want to place the train/test folders
# You don't need to create them!
test_folder = './data/mit67/test/'
train_folder = './data/mit67/train/'


def ignore_files(dir,files): return [f for f in files if os.path.isfile(os.path.join(dir,f))]

shutil.copytree(base_folder,test_folder,ignore=ignore_files)
shutil.copytree(base_folder,train_folder,ignore=ignore_files)


with open(train_path) as f:
    train_lines = f.readlines()

with open(test_path) as f:
    test_lines = f.readlines()

test_images, train_images = 0,0

for train_line in train_lines:
    train_line = train_line.strip() 
    if os.path.exists(base_folder+train_line):
        copyfile(base_folder+train_line, train_folder+train_line)
        train_images += 1

for test_line in test_lines:
    test_line = test_line.strip() 
    if os.path.exists(base_folder+test_line):
        copyfile(base_folder+test_line, test_folder+test_line)
        test_images += 1

print(train_images, test_images)
assert train_images == 67*80
assert test_images == 67*20

print('Dataset succesfully splitted!')
