import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import copy
import math
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import models, transforms, datasets

from models import get_backbone_class
from ssl import get_method_class
from util.imagenet_subset import ImageNetSubset
from util.merge_dataset import MergeDataset, MergeAllDataset
from util.transform import TwoCropTransform, MultiCropTransform, MAETransform
from util.sampling import get_selected_indices
from util.misc import *

DATASET_CONFIG = {'cars': 196, 'flowers': 102, 'pets': 37, 'aircraft': 100, 'cub': 200, 'dogs': 120, 'mit67': 67, 
                  'stanford40': 40, 'dtd': 47, 'celeba': 307, 'food11': 11, 'imagenet': 1000, 'inaturalist': 11,
                  'places': 365, 'coco': 21, 'webvision':1000, 'webfg': 496, 'everything': 4}


def parse_args():
    parser = argparse.ArgumentParser('argument for self-supervised pretraining')

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
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--merge_dataset', action='store_true', 
                        help='merge dataset1 and dataset2')
    parser.add_argument('--dataset1', type=str, default=None, 
                        help='first dataset when using merged dataset')
    parser.add_argument('--data_folder1', type=str, default=None, 
                        help='dataset1 path')
    parser.add_argument('--dataset2', type=str, default=None, 
                        help='second dataset when using merged dataset')
    parser.add_argument('--data_folder2', type=str, default=None, 
                        help='dataset2 path')
        
    # model & method
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--method', type=str, default='byol')
    # specific arguments for BYOL, SwAV, and DINO method
    parser.add_argument('--ema_scheduler', action='store_false', 
                        help='decide whether to update ema value')
    parser.add_argument('--initial_ema', type=float, default=0.996, 
                        help='initial value for exponential moving average')
    parser.add_argument('--temp_warmup_epochs', type=int, default=0, 
                        help='warmup epochs for teacher temperature')
    parser.add_argument('--initial_temp', type=float, default=0.04, 
                        help='initial teacher temperature value')
    parser.add_argument('--weight_decay_end', type=float, default=0, 
                        help='when using wd scheduler, the last value of wd when using scheduler')
    parser.add_argument('--clip_grad', type=float, default=0, 
                        help='value of gradient clipping')
    parser.add_argument('--prototypes', type=int, default=100, 
                        help='Number of prototypes dimension for SwAV')
    parser.add_argument('--freeze_prototypes', type=int, default=1, 
                        help='Number of epochs during which we keep the prototypes fixed for SwAV (origin code use iteration)')
    parser.add_argument('--freeze_last_layer', type=int, default=0, 
                        help='Number of epochs during which we keep the output layer fixed for DINO')
    parser.add_argument('--local_crops_number', 
                        type=int, default=0, help='Number of multi crop for SwAV and DINO')
    # specific arguments for MAE method
    parser.add_argument('--mask_ratio', type=float, default=0.75, 
                        help='ratio of random masking for tokens')
    parser.add_argument('--norm_pix_loss', action='store_true', 
                        help='apply normalization on target pixel values')

    # arguments for sampling
    parser.add_argument('--no_sampling', action='store_true',
                        help='vanilla selfsup without sampling')
    parser.add_argument('--sampling_method', default='simcore', type=str, choices=['random', 'simcore'],
                        help='strategy to sampling from openset')
    parser.add_argument('--retrieval_ckpt', default=None, type=str,
                        help='pretrained checkpoint for a retrieval model')
    parser.add_argument('--sampling_ratio', default=0.0, type=float,
                        help='proportion to sample from the dataset2, if 0.0 no sampling')
    parser.add_argument('--sampling_times', default=1, type=int,
                        help='how many times to sampling coreset')
    parser.add_argument('--cluster_num', default=100, type=int,
                        help='the centroid number of k-means clustering')
    parser.add_argument('--stop', action='store_true',
                        help='stopping criteion in sampling')
    parser.add_argument('--stop_thresh', default=0.95, type=float,
                        help='threshold for the stopping criterion')
    parser.add_argument('--patience', default=20, type=int,
                        help='patience value for stopping criterion; if large k used, lower this value')
    parser.add_argument('--from_ssl_official', action='store_true',
                        help='load from self-supervised imagenet-pretrained model (official PyTorch or top-conference papers)')

    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    # print scheduler information
    print('Learning rate cosine scheduler option is {}, and an initial lr is {}'.format(args.cosine, args.learning_rate))
    args.wd_scheduler = True if args.weight_decay_end else False
    print('Weight decay scheduler option is {}, and an initial and last value is {} and {}'.format(args.wd_scheduler, args.weight_decay, args.weight_decay_end if args.wd_scheduler else args.weight_decay))
    if args.method in ['byol', 'dino']:
        print('EMA scheduler option is {}, and an initial and last value is {} and {}'.format(args.ema_scheduler, args.initial_ema, 1.0 if args.ema_scheduler else 0.996))
    
    if args.merge_dataset:
        assert args.dataset1 is not None and args.dataset2 is not None, 'Specify two merging datasets'
        assert args.data_folder1 is not None and args.data_folder2 is not None, 'Specify two merging dataset dierctories'
        assert args.dataset1 != args.dataset2, 'Use two different datasets'
        args.dataset = args.dataset1  # to match the n_cls, does not need here !

    args.model_name = '{}_{}_pretrain_{}'.format(args.dataset, args.model, args.method)
    if args.from_ssl_official:
        args.model_name += '_from_ssl_official'
    if args.merge_dataset:
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
    elif args.dataset.startswith('imagenet_sub'):
        args.n_cls = 100 # dummy -> not important
    else:
        raise NotImplementedError

    return args


def get_dataset(args, dataset, data_folder, val=False):
    if dataset in DATASET_CONFIG or dataset.startswith('imagenet'):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise NotImplementedError
    normalize = transforms.Normalize(mean=mean, std=std)

    if not val:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=args.img_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # transforms.GaussianBlur(kernel_size=(3,3)), # require too much cost
            transforms.ToTensor(),
            normalize,
        ])
        train_transform = TwoCropTransform(train_transform)

        if args.local_crops_number:
            train_transform = MultiCropTransform(args.local_crops_number)

        if args.method == 'mae':
            train_transform = MAETransform(args.img_size)
    else:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            normalize])
                
    if dataset.startswith('imagenet_sub'):
        traindir = os.path.join(data_folder, 'train')
        train_dataset = ImageNetSubset('./util/subclass/{:s}.txt'.format(dataset),
                                        root=traindir,
                                        transform=train_transform)
    elif dataset == 'inaturalist':
        train_dataset = datasets.INaturalist(root=data_folder,
                                             version='2021_train_mini',
                                             transform=train_transform)
    elif dataset == 'everything':
        train_dataset = MergeAllDataset(transform=train_transform)

    elif dataset in DATASET_CONFIG:
        if dataset in ['imagenet', 'places', 'coco', 'webvision', 'webfg']:
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
    # for DINO-ViT and MAE-ViT
    elif args.optimizer == 'adamw':
        param_groups = get_params_groups(model.online_encoder)
        if args.method == 'dino':
            optimizer = optim.AdamW(param_groups)
        elif args.method == 'mae':
            optimizer = optim.AdamW(param_groups, betas=(0.9, 0.95))
    else:
        raise NotImplemented
        
    return model, optimizer


def set_loader(args):
    # construct data loader
    if args.merge_dataset:
        dataset1 = get_dataset(args, args.dataset1, args.data_folder1)
        dataset2 = get_dataset(args, args.dataset2, args.data_folder2)
        train_dataset = MergeDataset(dataset1, dataset2)
    else:
        train_dataset = get_dataset(args, args.dataset, args.data_folder)
    
    drop_last = True if args.method in ['moco', 'swav'] else False
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=drop_last)

    return train_loader


def set_loader_with_indices(selected_indices, args):
    # "train_*" denotes the training dataset for "train" function (val=False)
    train_dataset1 = get_dataset(args, args.dataset1, args.data_folder1)
    train_dataset2 = get_dataset(args, args.dataset2, args.data_folder2)
        
    # get the remained samples of dataset2
    remained_indices = torch.tensor(range(len(train_dataset2)))
    mask = torch.ones_like(remained_indices, dtype=torch.bool)
    mask[selected_indices] = False
    remained_indices = remained_indices[mask]

    # merged dataset for training
    train_dataset1 = MergeDataset(train_dataset1, torch.utils.data.Subset(train_dataset2, indices=selected_indices))    
    drop_last = True if args.method in ['moco', 'swav'] else False
    train_loader = torch.utils.data.DataLoader(
            train_dataset1, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=drop_last)

    return train_loader


def train(train_loader, model, optimizer, epoch, args, scaler=None):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.method == 'mae':  # original MAE uses a per iteration lr scheduler instead of per epoch
            adjust_lr_wd(args, optimizer, epoch + idx / len(train_loader))
        model.on_step_start()

        # depending on the type of transform
        if type(images) != list:  # MAE method uses a single-viewed batch
            img1, img2 = images.cuda(non_blocking=True), None
            bsz = img1.shape[0]
        else:
            img1, img2 = images[0].cuda(non_blocking=True), images[1].cuda(non_blocking=True)
            bsz = img1.shape[0]
            if args.method in ['swav', 'dino'] and args.local_crops_number:
                img1 = torch.cat([img1, img2], dim=0)
                img2 = [im.cuda(non_blocking=True) for im in images[2:]]
        
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
            if args.clip_grad:
                scaler.unscale_(optimizer)
                _ = clip_gradients(model.online_encoder, args.clip_grad)
            if args.method == 'swav': cancel_gradients_prototypes(epoch, model, args.freeze_prototypes)
            if args.freeze_last_layer: cancel_gradients_last_layer(epoch, model, args.freeze_last_layer)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad:
                _ = clip_gradients(model.online_encoder, args.clip_grad)
            if args.method == 'swav': cancel_gradients_prototypes(epoch, model, args.freeze_prototypes)
            if args.freeze_last_layer: cancel_gradients_last_layer(epoch, model, args.freeze_last_layer)
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

    if args.precision:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    model, optimizer = set_model(args)

    start_epoch = 1
    selected_indices = None
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] 
            start_epoch += 1
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

            if args.no_sampling:
                train_loader = set_loader(args)
            else:
                if 'indices' in checkpoint:
                    selected_indices = checkpoint['indices']
                    print("=> loaded selected indices length '{}'".format(len(selected_indices)))
                args = checkpoint['args']
                print(args)

                if args.stop and args.method == 'dino': 
                    model.temp_warmup_epochs = args.temp_warmup_epochs
                train_loader = set_loader_with_indices(selected_indices, args)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        if args.no_sampling:
            train_loader = set_loader(args)
        else:
            selected_indices, model, args = get_selected_indices(model, args)
            train_loader = set_loader_with_indices(selected_indices, args)
                
    model.train()
    for epoch in range(start_epoch, args.epochs+1):
        if args.method != 'mae':
            adjust_lr_wd(args, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        
        model.on_epoch_start(epoch)
        loss = train(train_loader, model, optimizer, epoch, args, scaler)
        model.on_epoch_end(epoch, args.epochs)

        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, loss {:.3f}'.format(epoch, time2-time1, loss))
        if epoch % args.save_freq == 0:
            save_file = os.path.join(args.save_folder, 'epoch_{}.pth'.format(epoch))
            save_model(model, optimizer, args, epoch, save_file, indices=selected_indices)

        if (not args.no_sampling) and (epoch in args.sampling_epochs): # only for args.sampling_times > 1
            print('renew the sampled coreset..')
            from util.sampling import random_sampling, simcore_sampling
            SAMPLING = {'random': random_sampling, 'simcore': simcore_sampling}

            selected_indices = SAMPLING[args.sampling_method](model, args)
            train_loader = set_loader_with_indices(selected_indices, args)

    save_file = os.path.join(args.save_folder, 'last.pth')
    save_model(model, optimizer, args, epoch, save_file)
          

if __name__ == '__main__':
    main()
