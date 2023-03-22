import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import math
import time
import random
import argparse
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets

from models import get_backbone_class
from util.semisup_dataset import ImageFolderSemiSup
from util.knn_evaluation import KNNValidation
from util.misc import *

DATASET_CONFIG = {'cars': 196, 'flowers': 102, 'pets': 37, 'aircraft': 100, 'cub': 200, 'dogs': 120, 'mit67': 67,
                  'stanford40': 40, 'dtd': 47, 'celeba': 307, 'food11': 11, 'imagenet': 1000}


def parse_args():
    parser = argparse.ArgumentParser('argument for supervised training')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='./save')
    parser.add_argument('--tag', type=str, default='',
                        help='tag for experiment name')
    
    # optimization
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
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

    # model & method
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='path to pre-trained model')
    parser.add_argument('--from_sl_official', action='store_true',
                        help='load from supervised imagenet-pretrained model (official PyTorch)')
    parser.add_argument('--from_ssl_official', action='store_true',
                        help='load from self-supervised imagenet-pretrained model (official PyTorch or top-conference papers)')
    
    # evaluation metric
    parser.add_argument('--e2e', action='store_true',
                        help='end-to-end finetuning')
    parser.add_argument('--knn', action='store_true',
                        help='k-NN evaluation (refer to Table 7a)')
    parser.add_argument('--topk', nargs='+', type=int, 
                        help='top-k value for k-NN evaluation')
    parser.add_argument('--label_ratio', type=float, default=1.0, 
                        help='ratio for the number of labeled sample (refer to Table 7b)')
    parser.add_argument('--multi_attribute', type=str, default='', 
                        help='multi-attribute setting for cars, aircraft, celeba dataset (refer to Table 7d)')

    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    args.wd_scheduler = False

    # for semi-supervised results
    args.semi = False
    if args.label_ratio != 1:
        print('For a semi-supervised training, follow SimCLR and BYOL protocols that finetune whole network')
        args.semi = True
        args.e2e = True
        # args.weight_decay = 0.0
        
    args.model_name = '{}_{}'.format(args.dataset, args.model)
    if args.e2e:
        args.model_name += '_e2e'
    else:
        # linear_evaluation
        args.model_name += '_le'

    if args.from_sl_official:
        assert 'resnet' in args.model or 'efficientnet' in args.model or 'timm' in args.model
        args.model_name += '_from_sl_official'
    elif args.from_ssl_official:
        args.model_name += '_from_ssl_official'
    else:
        if not args.pretrained:
            assert args.pretrained_ckpt is None
            args.model_name += '_from_scratch'
        else:
            if args.method:
                args.model_name += '_from_{}'.format(args.method)
            else:
                raise ValueError('Specify the pretrained method')

    if args.tag:
        args.model_name += '_{}'.format(args.tag)

    args.save_folder = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if args.warm:
        args.warmup_from = 0.01
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

    # for multi_attribute experiments
    if args.dataset == 'aircraft':
        if args.multi_attribute == 'family': args.n_cls = 70
        if args.multi_attribute == 'manufacturer': args.n_cls = 30
    if args.dataset == 'cars':
        if args.multi_attribute == 'type': args.n_cls = 9
        if args.multi_attribute == 'brand': args.n_cls = 49
    if args.dataset == 'celeba':
        if args.multi_attribute in ['oval', 'smiling', 'pointy', 'young']: args.n_cls = 2

    return args


def set_loader(args):
    # construct data loader
    if args.dataset in DATASET_CONFIG:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise NotImplementedError
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=args.img_size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(args.img_size),
                                        transforms.ToTensor(),
                                        normalize])

    if args.dataset in DATASET_CONFIG:
        if args.dataset == 'imagenet':
            traindir = os.path.join(args.data_folder, 'train') # under ~~/Data/CLS-LOC
            valdir = os.path.join(args.data_folder, 'val')
        else: # for fine-grained dataset
            if args.dataset == 'aircraft' or args.dataset == 'cars':
                traindir = os.path.join(args.data_folder, args.dataset, args.multi_attribute, 'train')
                valdir = os.path.join(args.data_folder, args.dataset, args.multi_attribute, 'test')
            elif args.dataset == 'celeba':
                traindir = os.path.join(args.data_folder, 'celeba_maskhq', args.multi_attribute, 'train')
                valdir = os.path.join(args.data_folder, 'celeba_maskhq', args.multi_attribute, 'test')
            else:
                traindir = os.path.join(args.data_folder, args.dataset, 'train')
                valdir = os.path.join(args.data_folder, args.dataset, 'test')
                
        if not args.semi:
            train_dataset = datasets.ImageFolder(root=traindir,
                                                 transform=train_transform)
        else:
            train_dataset = ImageFolderSemiSup(root=traindir,
                                               transform=train_transform,
                                               p=args.label_ratio)
        val_dataset = datasets.ImageFolder(root=valdir, transform=val_transform)
    else:
        raise NotImplementedError
    
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)

    return train_loader, val_loader


def set_model(args):    
    model = get_backbone_class(args.model)()
    feat_dim = model.final_feat_dim
    classifier = nn.Linear(feat_dim, args.n_cls)  # reset fc layer
    if args.method == 'mae':
        from models.dino_vit import trunc_normal_
        trunc_normal_(classifier.weight, std=0.01)
        classifier = torch.nn.Sequential(torch.nn.BatchNorm1d(feat_dim, affine=False, eps=1e-6), classifier)
        
        model.interpolate_pos_embed()
        model.set_mask_ratio(mask_ratio=0)
            
    criterion = nn.CrossEntropyLoss()

    if args.pretrained and args.pretrained_ckpt is not None:
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu')
        state_dict = ckpt['model']

        # HOTFIX: always dataparallel during pretraining
        new_state_dict = {}
        for k, v in state_dict.items():
            if "module." in k:
                k = k.replace("module.", "")
            if "backbone." in k:
                k = k.replace("backbone.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=False)
        print('pretrained model loaded from: {}'.format(args.pretrained_ckpt))

    if args.from_sl_official:
        if 'vit' not in args.model:
            model.load_sl_official_weights()
            print('pretrained model loaded from PyTorch ImageNet-pretrained')
        else:
            model = get_backbone_class(args.model)(pretrained=True)
            print('pretrained model loaded from Timm, and note that finetune IN-1k from IN-21k')
    
    if args.from_ssl_official:
        if args.model == 'resnet50':
            assert args.method == 'simclr'
            model.load_ssl_official_weights()
            print('pretrained model loaded from SimCLR ImageNet-pretrained official checkpoint')
        elif 'timm_dino' in args.model:
            assert args.method == 'dino'
            model = get_backbone_class(args.model)(pretrained=True)
            print('pretrained model via DINO loaded from Timm, and note that finetune IN-1k from IN-21k')
        else:
            raise NotImplemented

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    else:
        raise NotImplementedError
        
    model.cuda()
    classifier.cuda()
    criterion = criterion.cuda()
    
    optim_params = list(model.parameters()) + list(classifier.parameters()) if args.e2e else classifier.parameters()
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(optim_params,
                            lr=args.learning_rate,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay,
                            nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(optim_params,
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)
                            
    return model, classifier, criterion, optimizer


def train(train_loader, model, classifier, criterion, optimizer, epoch, args):
    model.train()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        if not args.e2e:
            with torch.no_grad():
                features = model(images)
            output = classifier(features.detach())
        else:
            output = classifier(model(images))
        loss = criterion(output, labels)

        losses.update(loss.item(), bsz)
        [acc1, acc5], _ = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, args, best_acc, best_model):
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = AverageMeter(), AverageMeter()
    if args.dataset in ['aircraft', 'pets', 'flowers', 'mit67']: # mean per-class accuracy
        top1, top5 = AverageClassMeter(args.n_cls), AverageClassMeter(args.n_cls)

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            output = classifier(model(images))
            loss = criterion(output, labels)

            losses.update(loss.item(), bsz)
            top1, top5 = update_metric(output, labels, top1, top5, args)
            batch_time.update(time.time() - end)
            end = time.time()
            if (idx + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.2f}, Acc@5 {top5.avg:.2f}'.format(top1=top1, top5=top5))
    best_acc, bool = get_best_acc(top1.avg, top5.avg, best_acc)
    if bool:
        best_model = deepcopy(model.state_dict())

    return best_acc, best_model


def main():
    args = parse_args()
    with open(os.path.join(args.save_folder, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    best_acc, best_model = [0, 0, 0], None
    model, classifier, criterion, optimizer = set_model(args)
    train_loader, val_loader = set_loader(args)
    
    # k-NN evaluation
    if args.knn:
        knn = KNNValidation(args)
        knn_acc = knn.topk_retrieval(model)
        best_acc += knn_acc
    
    for epoch in range(1, args.epochs+1):
        adjust_lr_wd(args, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion, optimizer, epoch, args)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2-time1, acc))
        best_acc[2] = acc.item()
        
        # eval for one epoch
        best_acc, best_model = validate(val_loader, model, classifier, criterion, args, best_acc, best_model)

        if epoch % args.save_freq == 0:
            save_file = os.path.join(args.save_folder, 'epoch_{}.pth'.format(epoch))
            save_model(model, optimizer, args, epoch, save_file)

    save_file = os.path.join(args.save_folder, 'best.pth')
    model.load_state_dict(best_model)
    save_model(model, optimizer, args, epoch, save_file)
    update_json('%s' % args.model_name, best_acc, path=os.path.join(args.save_dir, 'results.json'))


if __name__ == '__main__':
    main()
