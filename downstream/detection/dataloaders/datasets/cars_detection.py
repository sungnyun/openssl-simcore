import os
import cv2
import torch
import numpy as np
import pandas as pd
import scipy.io


class CarDetectionDataset(object):
    def __init__(self, root, split, transforms=None, class_label=False):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.class_label = class_label
        if not os.path.exists(os.path.join(self.root, '{}_bbox.csv'.format(self.split))):
            print('bbox csv file not exists > make...')
            self.df = self._make_bbox()
        else:
            self.df = pd.read_csv(os.path.join(self.root, '{}_bbox.csv'.format(self.split)))

        self.image_ids = self.df['img_path'].unique().tolist()
        
    def _make_bbox(self):
        train_bbox_path = os.path.join(self.root, 'train_bbox.csv')
        test_bbox_path = os.path.join(self.root, 'test_bbox.csv')

        metadata = scipy.io.loadmat(os.path.join(self.root, 'cars_annos.mat'))
        metadata = metadata['annotations'][0]
        train_bbox_list, test_bbox_list = list(), list()
        for meta in metadata:
            img_path = meta[0][0]
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, class_num, test = [meta[i][0,0] for i in range(1,7)]
            if not test:
                img_path = os.path.join(self.root, 'train', str(class_num), img_path.split('/')[1])
                train_bbox_list.append([img_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2, class_num - 1])
            else:
                img_path = os.path.join(self.root, 'test', str(class_num), img_path.split('/')[1])
                test_bbox_list.append([img_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2, class_num - 1])

        train_df = pd.DataFrame(train_bbox_list, columns=['img_path', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'labels'])
        test_df = pd.DataFrame(test_bbox_list, columns=['img_path', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'labels'])
        train_df.to_csv(train_bbox_path, index=False)
        test_df.to_csv(test_bbox_path, index=False)

        if self.split == 'train':
            return train_df
        elif self.split == 'test':
            return test_df

    def __len__(self):
        return len(self.image_ids)
        
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        records = self.df[self.df['img_path'] == image_id]
        image = cv2.imread(image_id, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].to_numpy()
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        if self.class_label:
            labels = records['labels'].to_numpy()
            labels = torch.tensor([labels[0]]).long()
        else:
            labels = torch.ones((records.shape[0],), dtype=torch.int64)
        
        # some images have wrong bbox labels, such as flipped bbox
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
