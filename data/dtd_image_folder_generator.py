import os
import scipy.io
import shutil
from shutil import copyfile


metadata = scipy.io.loadmat("./imdb/imdb.mat")
meta = metadata['images'][0,0]
ROOT = './images'

traindir = './train'
testdir = './test'

def ignore_files(dir,files): return [f for f in files if os.path.isfile(os.path.join(dir,f))]
shutil.copytree(ROOT, traindir, ignore=ignore_files)
shutil.copytree(ROOT, testdir, ignore=ignore_files)

trainlist = list()
vallist = list()
testlist = list()
with open('./labels/train1.txt') as f:
    for file in f.readlines():
        trainlist.append(file.strip())
with open('./labels/val1.txt') as f:
    for file in f.readlines():
        vallist.append(file.strip())
with open('./labels/test1.txt') as f:
    for file in f.readlines():
        testlist.append(file.strip())
        
trainlist = trainlist + vallist

train_images = 0
for img_path in trainlist:
    copyfile(os.path.join(ROOT, img_path), os.path.join(traindir, img_path))
    train_images += 1

test_images = 0
for img_path in testlist:
    copyfile(os.path.join(ROOT, img_path), os.path.join(testdir, img_path))
    test_images += 1

print(train_images,test_images)
assert train_images == 3760
assert test_images == 1880
print('Dataset succesfully splitted!')
