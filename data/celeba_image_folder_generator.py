import os
from shutil import copyfile

ROOT = './celeba_maskhq/'

orig_identities = {}
with open(os.path.join('./celeba', 'identity_CelebA.txt')) as f:
    lines = f.readlines()
    for line in lines:
        file_name, identity = line.strip().split()
        orig_identities[file_name] = identity

identities = {}

with open(os.path.join(ROOT, 'CelebA-HQ-to-CelebA-mapping.txt')) as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        if idx == 0: continue
        file_name, _, orig_file_name = line.strip().split()
        identities['{}.jpg'.format(file_name)] = orig_identities[orig_file_name]

print(f'There are {len(set(identities.values()))} identities.')
print(f'There are {len(identities.keys())} images.')

source_root = os.path.join(ROOT, 'CelebA-HQ-img')
target_root = os.path.join(ROOT, 'identity')
file_list = os.listdir(source_root)

for file in file_list:
    identity = identities[file]
    source = os.path.join(source_root, file)
    target = os.path.join(target_root, str(identity), file)
    if not os.path.exists(os.path.join(target_root, str(identity))):
        os.makedirs(os.path.join(target_root, str(identity)))
    copyfile(source, target)


# sample the identities with higher than 15 images

folder_root = os.path.join(ROOT, 'identity')
folder_list = os.listdir(folder_root)

threshold = 15
identity_cnt = 0

train_images = 0
test_images = 0
train_ratio = 0.8

for folder in folder_list:
    file_list = os.path.join(folder_root, folder)
    file_list = os.listdir(file_list)
    if len(file_list) >= threshold:
        identity_cnt += 1
        num_train = int(train_ratio * len(file_list))
        for file in file_list[:num_train]:
            train_images += 1
            source = os.path.join(folder_root, folder, file)
            target = os.path.join(folder_root, 'train', folder, file)
            if not os.path.exists(os.path.join(folder_root, 'train', folder)):
                os.makedirs(os.path.join(folder_root, 'train', folder))
            os.rename(source, target)
        for file in file_list[num_train:]:
            test_images += 1
            source = os.path.join(folder_root, folder, file)
            target = os.path.join(folder_root, 'test', folder, file)
            if not os.path.exists(os.path.join(folder_root, 'test', folder)):
                os.makedirs(os.path.join(folder_root, 'test', folder))
            os.rename(source, target)

print(f'There are {identity_cnt} identities that have more than {threshold} images.')
print(f'There are {train_images} train images.')
print(f'There are {test_images} test images.')


for folder in os.listdir(os.path.join(ROOT, 'identity')):
    if folder not in ['train', 'test']:
        os.system("rm -rf {}".format(os.path.join(ROOT, 'identity', folder)))

os.system("mv {} {}/".format(os.path.join(ROOT, 'identity', 'train'), ROOT))
os.system("mv {} {}/".format(os.path.join(ROOT, 'identity', 'test'), ROOT))
os.system("rm -rf {}".format(os.path.join(ROOT, 'identity')))


# For multi-attribute recognition for CelebA
# only use "Male" (20) and "Smiling" (31) attribute

multi_att = {}
with open(os.path.join(ROOT, 'CelebAMask-HQ-attribute-anno.txt')) as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        if idx == 0: continue
        elif idx == 1: 
            cls_to_att = line.strip().split()
            cls_to_att = {k: i for i, k in enumerate(cls_to_att)}
        else:
            attribute = line.strip().split()
            file_name, att_lst = attribute[0], attribute[1:]
            att_lst = [int(a.replace('-1', '0')) for a in att_lst]
            
            multi_att[file_name] = (att_lst[20], att_lst[31])

male, smile = 0, 0
for k, v in multi_att.items():
    if v[0] == 1: male +=1
    if v[1] == 1: smile += 1

print('whole CelebAMask-HQ dataset')

ATT = 'male'

os.makedirs(os.path.join(ROOT, '{}/train'.format(ATT)))
os.makedirs(os.path.join(ROOT, '{}/test'.format(ATT)))

for split in ['train', 'test']:
    source_root = os.path.join(ROOT, split)
    target_root = os.path.join(ROOT, ATT, split)
    folder_list = os.listdir(source_root)

    for folder in folder_list:
        for file in os.listdir(os.path.join(source_root, folder)):
            atts = multi_att[file]
            if ATT == 'male': att = atts[0]
            if ATT == 'smiling': att = atts[1]

            source = os.path.join(source_root, folder, file)
            target = os.path.join(target_root, str(att), file)

            if not os.path.exists(os.path.join(target_root, str(att))):
                os.makedirs(os.path.join(target_root, str(att)))
            copyfile(source, target)
