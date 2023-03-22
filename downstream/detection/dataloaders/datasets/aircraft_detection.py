import os
import cv2
import torch
import numpy as np
import pandas as pd


class AircraftDetectionDataset(object):
    def __init__(self, root, split, transforms=None, class_label=False):
        self.root = root
        self.split = split # 'trainval' or 'test'
        self.transforms = transforms
        self.class_label = class_label

        self.cls_to_idx = dict()
        self.img_to_cls = dict()
        num = 0
        with open(os.path.join(self.root, 'fgvc-aircraft-2013b/data', 'images_variant_{}.txt'.format(self.split)), 'r') as f:
            while True:
                line = f.readline()
                if not line: break
                line = line.strip().split(' ')
                img_id, cls_name = line[0]+'.jpg', '_'.join(line[1:]).replace('/', '_')
                self.img_to_cls[img_id] = cls_name
                if cls_name not in self.cls_to_idx:
                    self.cls_to_idx[cls_name] = num
                    num += 1
            f.close()

        if not os.path.exists(os.path.join(self.root, '{}_bbox.csv'.format(self.split))):
            print('bbox csv file not exists > make...')
            self.df = self._make_bbox()
        else:
            self.df = pd.read_csv(os.path.join(self.root, '{}_bbox.csv'.format(self.split)))

        self.image_ids = self.df['img_path'].unique().tolist()
        
    def _make_bbox(self):
        paths = self._split_data(self.split)

        images = []
        with open(os.path.join(self.root, 'fgvc-aircraft-2013b/data', 'images_box.txt'), 'r') as f:
            while True:
                line = f.readline()
                if not line: break

                line = line.replace('\n', '').split(' ')
                line[0] += '.jpg'

                if line[0] in paths:
                    line.append(int(self.cls_to_idx[self.img_to_cls[line[0]]]))
                    images.append(line)

        df = pd.DataFrame(np.array(images), columns=['img_path', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'labels'])
        
        # bbox type should be np.int64
        for i, col in enumerate(df.columns):
            if i != 0:
                df[col] = df[col].astype(np.int64)

        # save dataframe for ground_truth bbox later
        df.to_csv(os.path.join(self.root, '{}_bbox.csv'.format(self.split)), index=False)

        return df

    def _split_data(self, split):
        # split dataset to 'trainval' or 'test'        
        paths = []
        with open(os.path.join(self.root, 'fgvc-aircraft-2013b/data', 'images_variant_{}.txt'.format(split)), 'r') as f:
            while True:
                line = f.readline()
                if not line: break

                line = line.split(' ')[0] + '.jpg'
                paths.append(line)
            f.close()
        return paths

    def __len__(self):
        return len(self.image_ids)
        
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        records = self.df[self.df['img_path'] == image_id]
        image = cv2.imread(os.path.join(self.root, 'fgvc-aircraft-2013b/data/images', image_id), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].to_numpy()
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        if self.class_label:
            labels = records['labels'].to_numpy()
            labels = torch.tensor([labels[0]]).long()
        else:
            labels = torch.ones((records.shape[0],), dtype=torch.int64)
        
        # some images have wrong bbox labels, such as rotated bbox
        if not (boxes[:, 2] <= image.shape[1] and boxes[:, 0] >= 0 and boxes[:, 3] <= image.shape[0] and boxes[:, 1] >= 0):
            y1, x1, y2, x2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
            boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3] = x1, y1, x2, y2
            assert boxes[:, 2] <= image.shape[1] and boxes[:, 0] >= 0 and boxes[:, 3] <= image.shape[0] and boxes[:, 1] >= 0
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        target['area'] = torch.as_tensor(area, dtype=torch.float32)
        target['iscrowd'] = torch.zeros((records.shape[0],), dtype=torch.int64)
    
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
        
        return image.clone().detach(), target, image_id
