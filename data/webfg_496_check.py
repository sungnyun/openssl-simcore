import os

T_DIR = './WebFG-496/web-aircraft/train'
S_DIR = './aircraft/train'

num = 0
for folder in os.listdir(T_DIR):
    if folder.replace(' ', '_') in os.listdir(S_DIR):
        num += 1
    else:
        print(folder)

print(num)

os.system('mv {} {}'.format(os.path.join(T_DIR, 'F-16AB'), os.path.join(T_DIR, 'F-16A_B')))
os.system('mv {} {}'.format(os.path.join(T_DIR, 'FA-18'), os.path.join(T_DIR, 'F_A-18')))


T_DIR = './WebFG-496/web-cub/train'
S_DIR = './cub/train'

num = 0
for folder in os.listdir(T_DIR):
    if folder.replace(' ', '_') in os.listdir(S_DIR):
        num += 1
    else:
        print(folder)

print(num)
