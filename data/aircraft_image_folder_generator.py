import os
import shutil

ROOT = "./fgvc-aircraft-2013b/data"
TARGET_ROOT = "./"

LABEL = 'manufacturer'  # family, manufacturer

with open(os.path.join(ROOT, 'images_{}_trainval.txt'.format(LABEL))) as f:
    lines = f.readlines()

image_to_class = dict()
class_to_image = dict()
for line in lines:
    word = line.split(' ')
    imagename = word[0]
    classname = '_'.join(word[1:]).replace('/', '_').strip()
    image_to_class[imagename] = classname

SOURCE_DIR = os.path.join(ROOT, "./images")
TARGET_DIR = os.path.join(TARGET_ROOT, "./aircraft/{}/train".format(LABEL))

print("Copying images...")
for path, cls in image_to_class.items():
    source_path = os.path.join(SOURCE_DIR, path+'.jpg')
    target_path = os.path.join(TARGET_DIR, cls, path+'.jpg')
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copy(source_path, target_path)
print("ImageFolder directory created at {}".format(os.path.abspath(TARGET_DIR)))

with open(os.path.join(ROOT, 'images_{}_test.txt'.format(LABEL))) as f:
    lines = f.readlines()

image_to_class = dict()
class_to_image = dict()
for line in lines:
    word = line.split(' ')
    imagename = word[0]
    classname = '_'.join(word[1:]).replace('/', '_').strip()
    image_to_class[imagename] = classname
        
SOURCE_DIR = os.path.join(ROOT, "./images")
# TARGET_DIR = "./aircraft/test"
TARGET_DIR = os.path.join(TARGET_ROOT, "./aircraft/{}/test".format(LABEL))

print("Copying images...")
for path, cls in image_to_class.items():
    source_path = os.path.join(SOURCE_DIR, path+'.jpg')
    target_path = os.path.join(TARGET_DIR, cls, path+'.jpg')
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copy(source_path, target_path)
print("ImageFolder directory created at {}".format(os.path.abspath(TARGET_DIR)))
