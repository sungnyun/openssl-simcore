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
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms, datasets

from models import get_backbone_class
from util.methods import get_semisup_method_class
from util.dataset import ImageFolderSemiSup, x_u_split, MergeDataset
from util.transform import get_semisup_transform_class
from util.misc import *


DATASET_CONFIG = {'cars': (196, 8144), 'flowers': (102, 2040), 'pets': (37, 3680), 'aircraft': (100, 6667), 'cub': (200, 5990), 
                  'dogs': (120, 12000), 'mit67': (67, 5360), 'stanford40': (40, 4000), 'dtd': (47, 3760), 'celeba': (307, 4263), 
                  'food11': (11, 13296), 'imagenet100': (100, 130000), 'imagenet': (1000, 1281167)}

def parse_args():
    parser = argparse.ArgumentParser('argument for open-set semi-supervised learning')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./save')
    parser.add_argument('--tag', type=str, default='',
                        help='tag for experiment name')
    
    # optimization
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--learning_rate', type=float, default=0.03)
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
    parser.add_argument('--total_step', default=2**14, type=int,
                        help='number of total steps to run, SimCore (2**14) and scratch (2**16)')
    parser.add_argument('--eval_step', default=512, type=int,
                        help='number of eval steps to run')

    # dataset
    parser.add_argument('--dataset', type=str, default='cars')
    parser.add_argument('--data_folder', type=str, default='./data/')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=64,
                        help='only Self-Training with SimCore ckpt used 256')
    parser.add_argument('--label_ratio', type=float, default=0.5, 
                        help='ratio for the labeled samples')
    parser.add_argument('--mu', type=float, default=1, 
                        help='coefficient of unlabeled batch size, SelfTrain (1) and OpenMatch (2)')
    # for Self-Training
    parser.add_argument('--data_folder2', type=str, default='./data/',
                        help='data folder root for WebFG-496 dataset')
    parser.add_argument('--teacher_batch_size', type=int, default=64,
                        help='batch size for teacher_labeled_loader, SimCore (256) and scratch (64)')

    # model & method
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='path to pre-trained model')
    parser.add_argument('--teacher_pretrained_ckpt', type=str, default=None,
                        help='path to pre-trained teacher model')
    parser.add_argument('--use_ema', action='store_true',
                        help='whether to use exponential moving average of encoder model')
    parser.add_argument('--ema_decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--clip', default=0, type=float,
                        help='clip gradient value')

    parser.add_argument('--semisup_method', default='self_training', type=str,
                        help='semi-supervised learning algorithm')

    ### Self-Training ###
    parser.add_argument('--teacher_epochs', default=100, type=int,
                        help='epochs to train teacher network, SimCore (100) and scratch (500)')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--lambda_u', default=0.5, type=float,
                        help='coefficient of unlabeled loss')

    ### OpenMatch ###
    parser.add_argument('--start_fix', default=5, type=int,
                        help='epoch to start fixmatch training, SimCore (5) and scratch (20)')
    parser.add_argument('--lambda_oem', default=0.1, type=float,
                        help='coefficient of OEM loss')
    parser.add_argument('--lambda_socr', default=0.5, type=float,
                        help='coefficient of SOCR loss')
    # parser.add_argument('--T', default=1, type=float,
    #                     help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.0, type=float,
                        help='pseudo label threshold')

    args = parser.parse_args()

    args.epochs = math.ceil(args.total_step / args.eval_step)
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    args.wd_scheduler = False
        
    args.model_name = '{}_{}'.format(args.dataset, args.model)

    args.model_name += '_{}_semisup'.format(args.semisup_method)

    if not args.pretrained:
        assert args.pretrained_ckpt is None
        args.model_name += '_from_scratch'

    if args.tag:
        args.model_name += '_{}'.format(args.tag)

    args.save_folder = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if args.warm:
        args.warmup_from = args.learning_rate * 0.1**3
        args.warm_epochs = args.warm_epochs
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate

    if args.dataset in DATASET_CONFIG:
        args.n_cls, args.n_data = DATASET_CONFIG[args.dataset]
    else:
        raise NotImplementedError

    args.num_labeled = int(args.n_data * args.label_ratio)
    assert args.num_labeled >= args.batch_size

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

    # split labeled and unlabeled dataset
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args)
    TRANSFORM = get_semisup_transform_class(args.semisup_method)

    if args.dataset == 'imagenet':
        traindir = os.path.join(args.data_folder, 'train') # under ~~/Data/CLS-LOC
        valdir = os.path.join(args.data_folder, 'val')
    else: # for fine-grained dataset
        if args.dataset == 'aircraft' or args.dataset == 'cars':
            traindir = os.path.join(args.data_folder, args.dataset, 'train')
            valdir = os.path.join(args.data_folder, args.dataset, 'test')
        elif args.dataset == 'celeba':
            traindir = os.path.join(args.data_folder, 'celeba_maskhq', 'train')
            valdir = os.path.join(args.data_folder, 'celeba_maskhq', 'test')
        else:
            traindir = os.path.join(args.data_folder, args.dataset, 'train')
            valdir = os.path.join(args.data_folder, args.dataset, 'test')
    
    if args.semisup_method == 'openmatch':
        train_transform = TRANSFORM(args.img_size, mean, std)

    train_labeled_dataset = ImageFolderSemiSup(root=traindir,
                                                transform=train_transform,
                                                index=train_labeled_idxs)
    train_unlabeled_dataset = ImageFolderSemiSup(root=traindir,
                                                 transform=TRANSFORM(args.img_size, mean, std),
                                                 index=train_unlabeled_idxs)
                                                 
    traindir = os.path.join(args.data_folder2, 'train')
    webfg_ul_dataset = datasets.ImageFolder(root=traindir, transform=TRANSFORM(args.img_size, mean, std))
    train_unlabeled_dataset = MergeDataset(train_unlabeled_dataset, webfg_ul_dataset)
    
    val_dataset = datasets.ImageFolder(root=valdir, 
                                       transform=val_transform)

    if args.semisup_method == 'self_training':
        train_labeled_loader = [torch.utils.data.DataLoader(train_labeled_dataset, batch_size=args.batch_size, shuffle=True,
                                                            num_workers=8, pin_memory=True, drop_last=True),
                                torch.utils.data.DataLoader(train_labeled_dataset, batch_size=args.teacher_batch_size, shuffle=True,
                                                            num_workers=8, pin_memory=True, drop_last=True)]
    elif args.semisup_method == 'openmatch':
        train_labeled_loader = torch.utils.data.DataLoader(
                train_labeled_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=4, pin_memory=True, drop_last=True)
    
    train_unlabeled_loader = torch.utils.data.DataLoader(
            train_unlabeled_dataset, batch_size=int(args.batch_size*args.mu), shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True)

    return train_labeled_loader, train_unlabeled_loader, val_loader, args


def set_model(args):    
    model = get_backbone_class(args.model)()
    feat_dim = model.final_feat_dim
    classifier = nn.Linear(feat_dim, args.n_cls)  # reset fc layer
    criterion = nn.CrossEntropyLoss()
    semisup_criterion = get_semisup_method_class(args.semisup_method)(args)
    
    if args.pretrained_ckpt is not None:
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

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    else:
        raise NotImplementedError
        
    model.cuda()
    ema_model = ModelEMA(model, args.ema_decay) if args.use_ema else None

    classifier.cuda()
    criterion = criterion.cuda()
    semisup_criterion = semisup_criterion.cuda()
    optim_params = list(model.parameters()) + list(classifier.parameters())

    if args.semisup_method == 'openmatch':
        classifier_open = nn.Linear(feat_dim, args.n_cls*2, bias=False).cuda()
        optim_params += list(classifier_open.parameters())
        classifier = (classifier, classifier_open)
    
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
                            
    return model, ema_model, classifier, criterion, semisup_criterion, optimizer


def train_teacher(train_loader, model, classifier, criterion, optimizer, epoch, args):
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


def train_selftraining(train_labeled_loader, train_unlabeled_loader, model, ema_model, classifier, semisup_criterion, optimizer, scaler, epoch, args, teacher_model=None):
    model.train()
    classifier.train()

    batch_time, data_time = AverageMeter(), AverageMeter()
    l_losses, u_losses = AverageMeter(), AverageMeter()
    top1 = AverageMeter()

    labeled_iter = iter(train_labeled_loader)
    unlabeled_iter = iter(train_unlabeled_loader)
    
    end = time.time()
    for idx in range(args.eval_step):
        try:
            labeled_data = next(labeled_iter)
        except:
            labeled_iter = iter(train_labeled_loader)
            labeled_data = next(labeled_iter)
            
        try:
            unlabeled_data = next(unlabeled_iter)
        except:
            unlabeled_iter = iter(train_unlabeled_loader)
            unlabeled_data = next(unlabeled_iter)

        data_time.update(time.time() - end)
        
        warmup_learning_rate(args, epoch, idx, args.eval_step, optimizer)
        
        # semi-supervised learning methods
        epoch_info = [epoch, idx]
        with torch.cuda.amp.autocast():
            if teacher_model is None:
                loss, Lx, Lu, logits_x, targets_x = semisup_criterion(labeled_data, unlabeled_data, model, classifier, epoch_info)
            else: 
                loss, Lx, Lu, logits_x, targets_x = semisup_criterion(labeled_data, unlabeled_data, model, classifier, epoch_info, teacher_model)

        bsz = targets_x.shape[0]
        l_losses.update(Lx.item(), bsz)
        u_losses.update(Lu.item(), bsz)
        [acc1, acc5], _ = accuracy(logits_x, targets_x, topk=(1, 5))
        top1.update(acc1[0], bsz)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.clip)
            
        scaler.step(optimizer)
        scaler.update()
        if args.use_ema:
            ema_model.update(model)
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'l_loss {l_loss.val:.3f} ({l_loss.avg:.3f})\t'
                  'u_loss {u_loss.val:.3f} ({u_loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, args.eval_step, batch_time=batch_time, data_time=data_time,
                   l_loss=l_losses, u_loss=u_losses, top1=top1))
            sys.stdout.flush()

    return l_losses.avg, top1.avg


def exclude_dataset(args, dataset, model, classifier, classifier_open, exclude_known=False):
    end = time.time()
    # dataset.init_index()
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=True,
        shuffle=False)

    model.eval()
    classifier.eval()
    classifier_open.eval()
    with torch.no_grad():
        for batch_idx, ((_, _, inputs), targets) in enumerate(tqdm(test_loader)):
            inputs = inputs.cuda(non_blocking=True)
            feat = model(inputs)
            outputs, outputs_open = classifier(feat), classifier_open(feat)
            outputs = F.softmax(outputs, 1)
            out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
            tmp_range = torch.range(0, out_open.size(0) - 1).long().cuda()
            pred_close = outputs.data.max(1)[1]
            unk_score = out_open[tmp_range, 0, pred_close]
            known_ind = unk_score < 0.5
            if batch_idx == 0:
                known_all = known_ind
            else:
                known_all = torch.cat([known_all, known_ind], 0)
    known_all = known_all.data.cpu().numpy()
    if exclude_known:
        ind_selected = np.where(known_all == 0)[0]
    else:
        ind_selected = np.where(known_all != 0)[0]
    print("selected ratio %s"%( (len(ind_selected)/ len(known_all))))
    model.train()
    classifier.train()
    classifier_open.train()
    return ind_selected


def train_openmatch(train_labeled_loader, train_unlabeled_loader, model, ema_model, classifier, semisup_criterion, optimizer, scaler, epoch, args, teacher_model=None):
    model.train()
    classifier, classifier_open = classifier
    classifier.train()
    classifier_open.train()

    batch_time, data_time = AverageMeter(), AverageMeter()
    losses, losses_x = AverageMeter(), AverageMeter()
    losses_o, losses_oem, losses_socr, losses_fix = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    top1 = AverageMeter()

    labeled_iter = iter(train_labeled_loader)
    unlabeled_all_iter = iter(train_unlabeled_loader)

    unlabeled_dataset = deepcopy(train_unlabeled_loader.dataset)

    if epoch >= args.start_fix:
        indices_inlier = exclude_dataset(args, unlabeled_dataset, model, classifier, classifier_open)
    else:
        indices_inlier = range(len(unlabeled_dataset))

    unlabeled_sampler = torch.utils.data.SubsetRandomSampler(indices_inlier, generator=None)
    unlabeled_trainloader = torch.utils.data.DataLoader(
                            unlabeled_dataset, batch_size=int(args.batch_size*args.mu), sampler=unlabeled_sampler,
                            num_workers=4, pin_memory=True, drop_last=True)
    unlabeled_iter = iter(unlabeled_trainloader)
    
    end = time.time()
    for idx in range(args.eval_step):
        try:
            labeled_data = next(labeled_iter)
        except:
            labeled_iter = iter(train_labeled_loader)
            labeled_data = next(labeled_iter)
            
        try:
            unlabeled_data = next(unlabeled_iter)
        except:
            unlabeled_iter = iter(unlabeled_trainloader)
            unlabeled_data = next(unlabeled_iter)

        try:
            unlabeled_data_all = next(unlabeled_all_iter)
        except:
            unlabeled_all_iter = iter(train_unlabeled_loader)
            unlabeled_data_all = next(unlabeled_all_iter)

        data_time.update(time.time() - end)
        
        warmup_learning_rate(args, epoch, idx, args.eval_step, optimizer)
        
        # semi-supervised learning methods
        epoch_info = [epoch, idx]
        with torch.cuda.amp.autocast():
            loss, Lx, Lo, L_oem, L_socr, L_fix, logits_x, targets_x = semisup_criterion(labeled_data, unlabeled_data, unlabeled_data_all, model, classifier, classifier_open, epoch_info)

        bsz = targets_x.shape[0]
        losses.update(loss.item(), bsz)
        losses_x.update(Lx.item(), bsz)
        losses_o.update(Lo.item(), bsz)
        losses_oem.update(L_oem.item(), bsz)
        losses_socr.update(L_socr.item(), bsz)
        losses_fix.update(L_fix.item(), bsz)
        
        [acc1, acc5], _ = accuracy(logits_x, targets_x, topk=(1, 5))
        top1.update(acc1[0], bsz)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.clip)
            
        scaler.step(optimizer)
        scaler.update()
        if args.use_ema:
            ema_model.update(model)
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {losses.val:.3f} ({losses.avg:.3f})\t'
                  'loss_x {losses_x.val:.3f} ({losses_x.avg:.3f})\t'
                  'loss_o {losses_o.val:.3f} ({losses_o.avg:.3f})\t'
                  'loss_oem {losses_oem.val:.3f} ({losses_oem.avg:.3f})\t'
                  'loss_socr {losses_socr.val:.3f} ({losses_socr.avg:.3f})\t'
                  'loss_fix {losses_fix.val:.3f} ({losses_fix.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, args.eval_step, batch_time=batch_time, data_time=data_time,
                   losses=losses, losses_x=losses_x, losses_o=losses_o, losses_oem=losses_oem,
                   losses_socr=losses_socr, losses_fix=losses_fix, top1=top1))
            sys.stdout.flush()

    return losses_x.avg, top1.avg


def validate(val_loader, model, classifier, criterion, args, best_acc, best_model):
    model.eval()
    if isinstance(classifier, tuple):
        classifier = classifier[0]
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
        best_model = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict())]

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
    train_labeled_loader, train_unlabeled_loader, val_loader, args = set_loader(args)
    model, ema_model, classifier, criterion, semisup_criterion, optimizer = set_model(args)
    if args.semisup_method == 'self_training':
        model2, _, classifier2, _, _, optimizer2 = set_model(args)
    scaler = torch.cuda.amp.GradScaler()

    teacher_model = None
    if args.semisup_method == 'self_training':
        # for Self-Training, train a teacher network with labeled dataset
        train_labeled_loader, teacher_labeled_loader = train_labeled_loader
        if args.teacher_pretrained_ckpt is None:
            args.e2e = True
            for t_epoch in range(1, args.teacher_epochs+1):
                adjust_lr_wd(args, optimizer2, t_epoch)
                loss, acc = train_teacher(teacher_labeled_loader, model2, classifier2, criterion, optimizer2, t_epoch, args)

                if t_epoch % 5 == 0:
                    best_acc, best_model = validate(val_loader, model2, classifier2, criterion, args, best_acc, best_model)
            
            save_file = os.path.join(args.save_folder, 'teacher_best.pth')
            model2.load_state_dict(best_model[0])
            classifier2.load_state_dict(best_model[1])
            save_model(model2, optimizer2, args, t_epoch, save_file, classifier=classifier2)

            _, _ = validate(val_loader, model2, classifier2, criterion, args, best_acc, best_model)

            teacher_model = [model2, classifier2]
            best_acc, best_model = [0, 0, 0], None
        else:
            print('teacher pretrained model loaded from: {}'.format(args.teacher_pretrained_ckpt))
            ckpt = torch.load(args.teacher_pretrained_ckpt, map_location='cpu')
            model2.load_state_dict(ckpt['model'])
            classifier2.load_state_dict(ckpt['classifier'])
            teacher_model = [model2, classifier2]

            _, _ = validate(val_loader, model2, classifier2, criterion, args, best_acc, best_model)

        model2.eval()
        classifier2.eval()

    for epoch in range(1, args.epochs+1):
        adjust_lr_wd(args, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        if args.semisup_method == 'self_training':
            loss, acc = train_selftraining(train_labeled_loader, train_unlabeled_loader, model, ema_model, classifier, semisup_criterion, optimizer, scaler, epoch, args, teacher_model)
        elif args.semisup_method == 'openmatch':
            loss, acc = train_openmatch(train_labeled_loader, train_unlabeled_loader, model, ema_model, classifier, semisup_criterion, optimizer, scaler, epoch, args)
            
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2-time1, acc))
        best_acc[2] = acc.item()
        
        # eval for one epoch
        if args.use_ema:
            best_acc, best_model = validate(val_loader, ema_model.model, classifier, criterion, args, best_acc, best_model)
        else:
            best_acc, best_model = validate(val_loader, model, classifier, criterion, args, best_acc, best_model)
            
        if epoch % args.save_freq == 0:
            save_file = os.path.join(args.save_folder, 'epoch_{}.pth'.format(epoch))
            if args.use_ema:
                save_model(ema_model.model, optimizer, args, epoch, save_file)
            else:
                save_model(model, optimizer, args, epoch, save_file)

    save_file = os.path.join(args.save_folder, 'best.pth')
    model.load_state_dict(best_model[0])
    save_model(model, optimizer, args, epoch, save_file)
    update_json('%s' % args.model_name, best_acc, path=os.path.join(args.save_dir, 'results.json'))


if __name__ == '__main__':
    main()
