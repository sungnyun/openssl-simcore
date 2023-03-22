import os

dirs = 'dataset/WebVision'
cat = ['google', 'flickr']

os.makedirs(os.path.join(dirs, 'train'), exist_ok=True)

for c in cat:
    path = os.listdir(os.path.join(dirs, c))
    for p in path:
        if 'q' in p:
            tmp = '{}_{}'.format(c, p)
            os.system('mv {} {}'.format(os.path.join(dirs, c, p), os.path.joint(dirs, 'train', tmp)))
