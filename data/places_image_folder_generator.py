# this notebook is for places365 challenge
import os

ROOT = './places/data_256'
f = open(os.path.join('./places', 'categories_places365.txt'), 'r')

cls_lists = []
while True:
    line = f.readline()
    if not line: break
    
    path = line.split(' ')[0][1:]
    tmp = path.split('/')
    if len(tmp) == 3:
        tmp = '{}_{}'.format(tmp[1], tmp[2])
    elif len(tmp) == 2:
        tmp = '{}'.format(tmp[1])

    cls_lists.append((path, tmp))
f.close()

for p, t in cls_lists:
    path = os.path.join(ROOT, p)
    os.system('mv {}/ {}/'.format(path, os.path.join(ROOT, t)))

lst = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 
       'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'y', 'z']

for l in lst:
    os.system('rm -rf {}'.format(os.path.join(ROOT, l)))

path = os.listdir(os.path.join(ROOT))
os.system('mkdir {}'.format(os.path.join(ROOT, 'train')))

for p in path:
    os.system('mv {}/ {}/'.format(os.path.join(ROOT, p), os.path.join(ROOT, 'train')))

# rm data_256/train/airport_terminal/00020865.jpg
# rm data_256/train/airfield/00021054.jpg
# rm data_256/train/airport_terminal/00021622.jpg
# rm data_256/train/alley/00017667.jpg
# rm data_256/train/alley/00017729.jpg
# rm data_256/train/alley/00018289.jpg
# rm data_256/train/alley/00018892.jpg
# rm data_256/train/amusement_arcade/00007300.jpg

