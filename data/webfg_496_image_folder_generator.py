import os

dirs = './WebFG-496'
cat = ['web-aircraft', 'web-bird', 'web-car']

os.makedirs(os.path.join(dirs, 'train'), exist_ok=True)

for c in cat:
    for split in ['train']:
        path = os.listdir(os.path.join(dirs, c, split))
        
        for p in path:
            p_sp = p.replace(' ', '\ ')
            t = p.replace(' ', '_')
            source_dir = os.path.join(dirs, c, split, p_sp)
            target_dir = os.path.join(dirs, 'train')
            
            if t not in os.listdir(target_dir):
                os.system('mv {} {}'.format(source_dir, target_dir+'/{}'.format(t)))
            else:
                os.system('mv {} {}'.format(source_dir+'/*', os.path.join(target_dir, t)+'/'))


# there are 7 corrupted files

from PIL import Image
import os

path = "./WebFG-496/train"
folder = os.listdir(path)

for fol in folder:
    checkdir = os.path.join(path, fol)
    files = os.listdir(checkdir)
    format = [".jpg", ".jpeg"]

    for(p, dirs, f) in os.walk(checkdir):
        for file in f:
            if file.endswith(tuple(format)):
                try:
                    image = Image.open(p+"/"+file).load()
                    # print(image)
                except Exception as e:
                    print("An exception is raised:", e)
                    print(file)
