import os
import sys
import json
import math
import time
import random
import argparse
import warnings
warnings.filterwarnings('ignore')
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from models import get_backbone_class
from methods import get_method_class
from util.dataset import MergeDataset
from util.transform import TwoCropTransform
from util.misc import *

DATASET_CONFIG = {'cars': 196, 'flowers': 102, 'pets': 37, 'aircraft': 100, 'cub': 200, 'dogs': 120, 
                  'mit67': 67, 'stanford40': 40, 'dtd': 47, 'celeba': 307, 'food11': 11, 'imagenet100': 100, 
                  'imagenet': 1000, 'inaturalist': 11, 'places_standard': 365, 'places_challenge': 365, 'coco': 21}


def parse_args():
    parser = argparse.ArgumentParser('argument for self-supervised training with hard negative mining')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='./save')
    parser.add_argument('--tag', type=str, default='',
                        help='tag for experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='path of model checkpoint to resume')
    
    # optimization
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--lr_decay_epochs', type=str, default='200,300')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--precision', action='store_true',
                        help='mixed precision')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--warm_epochs', type=int, default=0,
                        help='warmup epochs')

    # dataset
    parser.add_argument('--dataset', type=str, default='cars')
    parser.add_argument('--data_folder', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--dataset1', type=str, default=None, 
                        help='first dataset when using merged dataset')
    parser.add_argument('--data_folder1', type=str, default=None, 
                        help='dataset1 path')
    parser.add_argument('--dataset2', type=str, default=None, 
                        help='second dataset when using merged dataset')
    parser.add_argument('--data_folder2', type=str, default=None, 
                        help='dataset2 path')
        
    # model & method
    parser.add_argument('--model', type=str, default='resnet18')
    
    # arguments for mining
    parser.add_argument('--method', default='sampling', type=str, choices=['sampling', 'explicit_sampling', 'explicit_sampling_memory'],
                        help='strategy to mining relevant samples')
    # for HardSampling method
    parser.add_argument('--beta', default=1.0, type=float,
                        help='hardness hyperparameter (bigger is harder)')
    parser.add_argument('--tau_plus', default=0.01, type=float,
                        help='assumption of pseudo class probability')
    # for ExpHardSampling method
    parser.add_argument('--sampling_ratio', default=0.7, type=float,
                        help='sampling ratio for top-similar instances')
    parser.add_argument('--simclr_temperature', default=0.1, type=float,
                        help='use different temperature value depending on the sampling ratio')

    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    # print scheduler information
    print('Learning rate cosine scheduler option is {}, and an initial lr is {}'.format(args.cosine, args.learning_rate))
    args.wd_scheduler = False
    print('Weight decay initial value is {}'.format(args.weight_decay))

    assert args.dataset1 is not None and args.dataset2 is not None, 'Specify two merging datasets'
    assert args.data_folder1 is not None and args.data_folder2 is not None, 'Specify two merging dataset dierctories'
    assert args.dataset1 != args.dataset2, 'Use two different datasets'
    args.dataset = args.dataset1  # to match the n_cls, does not need here !

    args.model_name = '{}_{}_pretrain_{}'.format(args.dataset, args.model, args.method)
    args.model_name += '_merge_{}'.format(args.dataset2)
    if args.tag:
        args.model_name += '_{}'.format(args.tag)
    args.save_folder = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if args.warm:
        args.warmup_from = args.learning_rate * 0.1
        if not args.warm_epochs:
            args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate

    if args.dataset in DATASET_CONFIG:
        args.n_cls = DATASET_CONFIG[args.dataset]
    else:
        raise NotImplementedError

    return args


def get_dataset(args, dataset, data_folder, twocrop=True):
    if dataset in DATASET_CONFIG:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise NotImplementedError
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=args.img_size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        # transforms.GaussianBlur(kernel_size=(3,3)), # require too much cost
        transforms.ToTensor(),
        normalize,
    ])
    if twocrop:
        train_transform = TwoCropTransform(train_transform)
                
    if dataset in DATASET_CONFIG:
        if dataset in ['imagenet', 'places_standard', 'places_challenge', 'coco']:
            traindir = os.path.join(data_folder, 'train')
        elif dataset == 'celeba': # use mini-version of CelebAMask_HQ
            traindir = os.path.join(data_folder, 'celeba_maskhq', 'train')
        else:
            traindir = os.path.join(data_folder, dataset, 'train')
        train_dataset = datasets.ImageFolder(root=traindir,
                                             transform=train_transform)
    else:
        raise NotImplementedError

    return train_dataset 


def set_loader(args):
    # construct data loader
    if args.method == 'sampling' or args.method == 'explicit_sampling':
        dataset1 = get_dataset(args, args.dataset1, args.data_folder1)
        dataset2 = get_dataset(args, args.dataset2, args.data_folder2)
        train_dataset = MergeDataset(dataset1, dataset2)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True)
        return train_loader, None

    elif args.method == 'explicit_sampling_memory':
        dataset1 = get_dataset(args, args.dataset1, args.data_folder1)
        dataset2 = get_dataset(args, args.dataset2, args.data_folder2, twocrop=False)
        train_loader1 = torch.utils.data.DataLoader(
            dataset1, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)
        train_loader2 = torch.utils.data.DataLoader(
            dataset2, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True)
        return train_loader1, train_loader2


def set_model(args):
    backbone = get_backbone_class(args.model)()
    model = get_method_class(args.method)(backbone, args)

    if torch.cuda.device_count() > 1:
        model._data_parallel()
    model.cuda()
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), 
                               lr=args.learning_rate, 
                               weight_decay=args.weight_decay)
        print('Using Adam optimizer...')
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay,
                              nesterov=True)
    else:
        raise NotImplemented
    return model, optimizer


def train(train_loader, train_loader2, model, optimizer, epoch, args, scaler=None):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        model.on_step_start(train_loader2)

        # depending on the type of transform
        img1, img2 = images[0].cuda(non_blocking=True), images[1].cuda(non_blocking=True)
        bsz = img1.shape[0]
        
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)
        if args.precision:
            with torch.cuda.amp.autocast():
                loss = model.compute_ssl_loss(img1, img2)
        else:
            loss = model.compute_ssl_loss(img1, img2)
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        if args.precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        model.on_step_end()
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    args = parse_args()
    with open(os.path.join(args.save_folder, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    model, optimizer = set_model(args)
    train_loader1, train_loader2 = set_loader(args)
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] 
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch += 1
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 1

    if args.precision:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    for epoch in range(args.start_epoch, args.epochs+1):
        adjust_lr_wd(args, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        
        model.on_epoch_start(train_loader2)
        loss = train(train_loader1, train_loader2, model, optimizer, epoch, args, scaler)
        model.on_epoch_end()

        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, loss {:.3f}'.format(epoch, time2-time1, loss))
    
        if epoch % args.save_freq == 0:
            save_file = os.path.join(args.save_folder, 'epoch_{}.pth'.format(epoch))
            save_model(model, optimizer, args, epoch, save_file)

    save_file = os.path.join(args.save_folder, 'last.pth')
    save_model(model, optimizer, args, epoch, save_file)
          

if __name__ == '__main__':
    main()
