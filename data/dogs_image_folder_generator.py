from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import os
import shutil
import scipy.io
from shutil import copyfile

# Change the path to your dataset folder:
base_folder = './Images/'

train_path = './train_list.mat'
test_path = './test_list.mat'
train_metadata = scipy.io.loadmat(train_path)['file_list']
test_metadata = scipy.io.loadmat(test_path)['file_list']

# Here declare where you want to place the train/test folders
# You don't need to create them!
test_folder = './data/dogs/test/'
train_folder = './data/dogs/train/'


def ignore_files(dir,files): return [f for f in files if os.path.isfile(os.path.join(dir,f))]

shutil.copytree(base_folder,test_folder,ignore=ignore_files)
shutil.copytree(base_folder,train_folder,ignore=ignore_files)

train_lines, test_lines = [], []
for i in range(len(train_metadata)):
    train_lines.append(train_metadata[i][0][0])
for i in range(len(test_metadata)):
    test_lines.append(test_metadata[i][0][0])

test_images, train_images = 0,0

for train_line in train_lines:
    if os.path.exists(base_folder+train_line):
        copyfile(base_folder+train_line, train_folder+train_line)
        train_images += 1

for test_line in test_lines:
    if os.path.exists(base_folder+test_line):
        copyfile(base_folder+test_line, test_folder+test_line)
        test_images += 1

print(train_images, test_images)
assert train_images == 12000
assert test_images == 8580

print('Dataset succesfully splitted!')
