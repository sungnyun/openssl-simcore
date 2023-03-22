import os
import shutil

with open('./annotations/trainval.txt') as f:
    lines = f.readlines()

image_to_class = dict()
for line in lines:
    if line.startswith('#'):
        continue
    word = line.split(' ')
    image_to_class[word[0]] = word[1]
# image_to_class

SOURCE_DIR = "./images"
TARGET_DIR = "./pets/train"
print("Copying images...")
for path, cls in image_to_class.items():
    source_path = os.path.join(SOURCE_DIR, path+'.jpg')
    target_path = os.path.join(TARGET_DIR, cls, path+'.jpg')
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copy(source_path, target_path)
print("ImageFolder directory created at {}".format(os.path.abspath(TARGET_DIR)))

with open('./annotations/test.txt') as f:
    lines = f.readlines()

image_to_class = dict()
for line in lines:
    if line.startswith('#'):
        continue
    word = line.split(' ')
    image_to_class[word[0]] = word[1]

SOURCE_DIR = "./images"
TARGET_DIR = "./pets/test"
print("Copying images...")
for path, cls in image_to_class.items():
    source_path = os.path.join(SOURCE_DIR, path+'.jpg')
    target_path = os.path.join(TARGET_DIR, cls, path+'.jpg')
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copy(source_path, target_path)
print("ImageFolder directory created at {}".format(os.path.abspath(TARGET_DIR)))

