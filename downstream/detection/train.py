import os
import sys
import math
import shutil
import argparse
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataloaders import make_data_loader

from util.misc import adjust_learning_rate, save_model


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    
    parser.add_argument('--save_dir', type=str, default='./save')
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='aircraft', choices=['aircraft', 'cars'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--tag', type=str, default='',
                        help='tag for experiment name')
    parser.add_argument('--root', type=str, default='./',
                        help='data root')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50'],
                        help='backbone name (only implement resnet50)')
    parser.add_argument('--pretrained_backbone', type=str, default='',
                        help='pretrained path for only backbone network')
    parser.add_argument('--resume', type=str, default=None,
                        help='path of model checkpoint to resume')
    parser.add_argument('--pretrained_ckpt', type=str, default='',
                        help='pretrained path for the entire detection models')
    parser.add_argument('--class_label', action='store_true', default=False,
                        help='use class labels or not')

    # optimization                        
    parser.add_argument('--precision', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--lr_decay_epochs', type=str, default='4,8')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')

    # prediction
    parser.add_argument('--iou_thresh', type=float, default=0.5,
                        help='iou threshold for nms operation')
    parser.add_argument('--predict_path', type=str, default='./mAP/input/detection-results',
                        help='path for prediction of bbox')
    parser.add_argument('--delete_prev_pred', action='store_false',
                        help='decide whether to delete previous prediction results')
    parser.add_argument('--gt_bbox_path', type=str, default='./mAP/input/ground-truth',
                        help='path for ground truth of bbox')
    
    args = parser.parse_args()
    
    batch_sizes = {'cars': 16, 'aircraft': 16}
    args.batch_size = batch_sizes[args.dataset]

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    args.save_file = os.path.join(args.save_dir, '{}_{}'.format(args.dataset, args.tag))
    os.makedirs(args.save_file, exist_ok=True)

    return args


def get_model(num_class, pretrained_backbone=None):
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, 
    #                                                              pretrained_backbone=False, 
    #                                                              trainable_backbone_layers=0,
    #                                                              min_size=200,
    #                                                              max_size=333)
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_class)
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=False,
                                                                pretrained_backbone=False,
                                                                trainable_backbone_layers=0,
                                                                min_size=200,
                                                                max_size=333)
                                                                # num_classes=num_class,
    # when pretrained=True (load pretrained fpn on COCO)
    # in_features = 256
    # model.head.classification_head.num_classes = num_class
    # model.head.classification_head.cls_logits = nn.Conv2d(in_features, num_class*9, kernel_size=3, stride=1, padding=1)

    # load pretrained backbone, not for FPN
    if pretrained_backbone:
        assert os.path.exists(pretrained_backbone)
        ckpt = torch.load(pretrained_backbone, map_location='cpu')
        state_dict = ckpt['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            if "head." in k or "classifier." in k:
                continue
            if "module." in k:
                k = k.replace("module.", "")
            if "backbone." in k:
                k = k.replace("backbone.", "")
            new_state_dict["body." + k] = v
        state_dict = new_state_dict
        missing, unexpected = model.backbone.load_state_dict(state_dict, strict=False)
        print('Missing keys: {}'.format(missing))
        print('Unexpected keys: {}'.format(unexpected))
        print('pretrained model loaded from: {}'.format(pretrained_backbone))

    # freeze backbone
    modules = model.backbone.body
    for name, m in modules.named_modules():
        for p in m.parameters():
            p.requires_grad = False

    return model


def train(model, optimizer, train_loader, epoch, scaler, args):
    model.train()
        
    all_losses = []
    all_losses_dict = []
    
    for images, targets, _ in tqdm(train_loader):
        # torchvision faster-rcnn gets list-type inputs
        images = list(image.cuda() for image in images)
        targets = [{k: torch.tensor(v).cuda() for k, v in t.items()} for t in targets]

        if args.precision:
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
        else:
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()
        
        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)
        
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping trainig") 
            print(loss_dict)
            sys.exit(1)
        
        optimizer.zero_grad()
        if args.precision:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()        
        
    all_losses_dict = pd.DataFrame(all_losses_dict)
    # print("Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
    #     epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
    #     all_losses_dict['loss_classifier'].mean(),
    #     all_losses_dict['loss_box_reg'].mean(),
    #     all_losses_dict['loss_rpn_box_reg'].mean(),
    #     all_losses_dict['loss_objectness'].mean()
    # ))
    print("Epoch {}, lr: {:.6f}, loss: {:.6f}, classification: {:.6f}, bbox_regression: {:.6f}".format(
        epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
        all_losses_dict['classification'].mean(),
        all_losses_dict['bbox_regression'].mean()))


    # save model
    if epoch % args.save_freq == 0:
        save_model(model, optimizer, args, epoch, os.path.join(args.save_file, 'epoch_{}.pth'.format(epoch)))


def predict_bbox(model, test_set, args):
    def apply_nms(orig_prediction, iou_thresh=0.5):
        keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
        
        final_prediction = orig_prediction
        final_prediction['boxes'] = final_prediction['boxes'][keep]
        final_prediction['scores'] = final_prediction['scores'][keep]
        final_prediction['labels'] = final_prediction['labels'][keep]
        
        return final_prediction

    model.eval()
    with torch.no_grad():
        nms_predictions = list()
        for idx in tqdm(range(len(test_set))):
            img, target, img_id = test_set[idx]
            prediction = model([img.cuda()])[0]

            nms_prediction = apply_nms(prediction, iou_thresh=args.iou_thresh)
            nms_prediction['image_id'] = img_id
            nms_predictions.append(nms_prediction)

    predict_path = os.path.join(args.predict_path, args.dataset)
    if args.delete_prev_pred and os.path.isdir(predict_path):
        shutil.rmtree(predict_path)
        print("delete previous prediction results in '{}'".format(predict_path))
    os.makedirs(predict_path, exist_ok=True)
    print("make directory for prediction results in '{}'".format(predict_path))
    
    for prediction in nms_predictions:
        img_id = prediction['image_id'].split('/')[-1].replace('.jpg','')
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()

        with open(os.path.join(predict_path, '{}.txt'.format(img_id)), 'w') as f:
            for i in range(len(scores)):
                confidence = scores[i]
                left = round(boxes[i,0])
                top = round(boxes[i,1])
                right = round(boxes[i,2])
                bottom = round(boxes[i,3])
                label = int(labels[i]) if args.class_label else 0
                f.write('{}{:d} {:.6f} {:d} {:d} {:d} {:d}\n'.format(args.dataset, label, confidence, left, top, right, bottom))
            f.close()
    print("save prediction to '{}'".format(predict_path))


def save_gt_bbox(args):
    gt_path = os.path.join(args.gt_bbox_path, args.dataset)
    if args.delete_prev_pred and os.path.isdir(gt_path):
        shutil.rmtree(gt_path)
        print("delete previous ground truth bbox in '{}'".format(gt_path))
    os.makedirs(gt_path, exist_ok=True)
    print("make directory for ground truth bbox '{}'".format(gt_path))

    test_metadata = pd.read_csv(os.path.join(args.root, 'test_bbox.csv'))
    for i in range(len(test_metadata)):
        img_path, x1, y1, x2, y2, label = test_metadata.iloc[i, :].values
        img_id = img_path.split('/')[-1].replace('.jpg','')
        with open(os.path.join(gt_path, '{}.txt'.format(img_id)), 'w') as f:
            left = int(x1)
            top = int(y1)
            right = int(x2)
            bottom = int(y2)
            f.write('{}{:d} {:d} {:d} {:d} {:d}\n'.format(args.dataset, int(label), left, top, right, bottom))
            f.close()


def main():
    args = parse_args()

    train_loader, test_set, num_class = make_data_loader(args)
    if args.precision:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if args.pretrained_ckpt:
        model = get_model(num_class)
        checkpoint = torch.load(args.pretrained_ckpt)
        model.load_state_dict(checkpoint['model'])
        model = model.cuda()
    else:
        model = get_model(num_class, args.pretrained_backbone)
        model = model.cuda()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
        
        start_epoch = 1
        for epoch in range(start_epoch, args.epochs+1):
            adjust_learning_rate(args, optimizer, epoch)
            train(model, optimizer, train_loader, epoch, scaler, args)

        save_model(model, optimizer, args, epoch, os.path.join(args.save_file, 'last.pth'))

    predict_bbox(model, test_set, args)
    save_gt_bbox(args)

if __name__ == "__main__":
    main()
