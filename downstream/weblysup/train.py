import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import time
import random
import argparse
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets

from models import get_backbone_class
from util.dataset import IndexDataset, MergeDataset, NoisyMergeDataset
from util.methods import CoTeaching, DivideMix
from util.misc import *

DATASET_CONFIG = {'cars': 196, 'flowers': 102, 'pets': 37, 'aircraft': 100, 'cub': 200, 'dogs': 120, 'mit67': 67,
                  'stanford40': 40, 'dtd': 47, 'celeba': 307, 'food11': 11, 'imagenet100': 100, 'imagenet': 1000}


def parse_args():
    parser = argparse.ArgumentParser('argument for webly-supervised learning')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='./save')
    parser.add_argument('--tag', type=str, default='',
                        help='tag for experiment name')
    
    # optimization
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.03)
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')

    # dataset
    parser.add_argument('--dataset', type=str, default='aircraft')
    parser.add_argument('--data_folder', type=str, default='./data/')
    parser.add_argument('--data_folder2', type=str, default='./data/')
    parser.add_argument('--label_ratio', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=256,
                        help='path to pre-trained model, Co-Teaching (256) and DivideMix (128)')

    # model & method
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--noise_method', type=str, default='co_teaching')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='path to pre-trained model')
    # Co-teaching
    parser.add_argument('--forget_rate', type=float, default=0.2,
                        help='used 0.1 value only for Cars dataset')
    parser.add_argument('--num_gradual', type=int, default=10, 
                        help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-Teaching paper.')
    parser.add_argument('--exponent', type=float, default=1, 
                        help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-Teaching paper.')
    # DivideMix
    parser.add_argument('--warmup', type=int, default=30,
                        help='warmup epochs for DivideMix algorithm, only training from scratch on Cars used 60 epochs')
    parser.add_argument('--alpha', default=0.75, type=float, 
                        help='parameter for Beta')
    parser.add_argument('--lambda_u', default=75, type=float, 
                        help='weight for unsupervised loss')
    parser.add_argument('--p_threshold', default=0.2, type=float, 
                        help='clean probability threshold')
    parser.add_argument('--T', default=0.5, type=float, 
                        help='sharpening temperature')

    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    args.wd_scheduler = False
        
    args.model_name = '{}_{}'.format(args.dataset, args.model)
    args.model_name += '_{}_noise'.format(args.noise_method)

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

    if args.dataset in DATASET_CONFIG:
        args.n_cls = DATASET_CONFIG[args.dataset]
    elif args.dataset.startswith('imagenet_sub'):
        args.n_cls = 100 # dummy -> not important
    else:
        raise NotImplementedError

    return args


def set_loader(args):
    def _get_dir(dataset, data_folder):
        if dataset == 'imagenet':
            traindir = os.path.join(data_folder, 'train') # under ~~/Data/CLS-LOC
            valdir = os.path.join(data_folder, 'val')
        else: # for fine-grained dataset
            if dataset == 'celeba':
                traindir = os.path.join(data_folder, 'celeba_maskhq', 'train')
                valdir = os.path.join(data_folder, 'celeba_maskhq', 'test')
            else:
                traindir = os.path.join(data_folder, dataset, 'train')
                valdir = os.path.join(data_folder, dataset, 'test')
        return traindir, valdir

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
                
    traindir1, valdir = _get_dir(args.dataset, args.data_folder)
    traindir2, _ = _get_dir('web-'+args.dataset, args.data_folder2)

    train_dataset1 = datasets.ImageFolder(root=traindir1,
                                          transform=train_transform)
    train_dataset2 = datasets.ImageFolder(root=traindir2,
                                          transform=train_transform)
    
    merge_dataset = MergeDataset(train_dataset1, train_dataset2)
    val_dataset = datasets.ImageFolder(root=valdir, transform=val_transform)
    
    train_loader = torch.utils.data.DataLoader(
            merge_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)

    return train_loader, val_loader, merge_dataset, train_dataset1, train_dataset2


def set_model(args):    
    model = get_backbone_class(args.model)()
    feat_dim = model.final_feat_dim
    classifier = nn.Linear(feat_dim, args.n_cls)  # reset fc layer            
    
    if args.noise_method == 'co_teaching':
        criterion = CoTeaching()
    elif args.noise_method == 'dividemix':
        criterion = DivideMix(lambda_u=args.lambda_u)

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

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    else:
        raise NotImplementedError
        
    model.cuda()
    classifier.cuda()
    criterion = criterion.cuda()
    
    optim_params = list(model.parameters()) + list(classifier.parameters())
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


def train_coteaching(train_loader, model1, model2, classifier1, classifier2, criterion, optimizer1, optimizer2, epoch, rate_schedule, scaler, args):
    model1.train()
    classifier1.train()
    model2.train()
    classifier2.train()

    batch_time, data_time = AverageMeter(), AverageMeter()
    losses, top1 = AverageMeter(), AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
    
        with torch.cuda.amp.autocast():
            output1 = classifier1(model1(images))
            output2 = classifier2(model2(images))

            loss1, loss2 = criterion(output1, output2, labels, rate_schedule[epoch-1])

        losses.update(loss1.item(), bsz)
        [acc1, acc5], _ = accuracy(output1, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        optimizer1.zero_grad()
        scaler.scale(loss1).backward()            
        scaler.step(optimizer1)
        scaler.update()

        optimizer2.zero_grad()
        scaler.scale(loss2).backward()            
        scaler.step(optimizer2)
        scaler.update()
    
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


def train_dividemix_warmup(train_loader, model, classifier, criterion, optimizer, epoch, scaler, args):
    model.train()
    classifier.train()

    batch_time, data_time = AverageMeter(), AverageMeter()
    losses, top1 = AverageMeter(), AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        with torch.cuda.amp.autocast():
            output = classifier(model(images))
            loss = criterion(output, labels)

        losses.update(loss.item(), bsz)
        [acc1, acc5], _ = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        optimizer.zero_grad()
        scaler.scale(loss).backward()            
        scaler.step(optimizer)
        scaler.update()
    
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


def train_dividemix(merge_dataset, train_dataset1, train_dataset2, model1, model2, classifier1, classifier2, criterion, optimizer1, optimizer2, epoch, scaler, args):
    def co_divide(train_dataset1, train_dataset2, pred, clean_prob):        
        pred_idx = pred.nonzero()[0]
        labeled_dataset = NoisyMergeDataset(train_dataset1, train_dataset2, clean_prob, index=pred_idx, return_prob=True)

        pred_idx = (1 - pred).nonzero()[0]
        unlabeled_dataset = NoisyMergeDataset(train_dataset1, train_dataset2, clean_prob, index=pred_idx, return_prob=False)

        if len(labeled_dataset) and len(unlabeled_dataset):
            labeled_loader = torch.utils.data.DataLoader(
                labeled_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True)
            unlabeled_loader = torch.utils.data.DataLoader(
                unlabeled_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True)
            return labeled_loader, unlabeled_loader, True
        else:
            return None, None, False

    def mixmatch_train(labeled_loader, unlabeled_loader, net, net2, classifier, classifier2, criterion, optimizer, epoch, args):
        losses, top1 = AverageMeter(), AverageMeter()

        net.train()
        classifier.train()
        net2.eval() # fix one network and train the other
        classifier2.eval()
        
        labeled_train_iter = iter(labeled_loader)
        unlabeled_train_iter = iter(unlabeled_loader)    
        num_iter = (len(labeled_loader.dataset) // args.batch_size) + 1
        # for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_loader):      
        for batch_idx in range(len(labeled_loader)):
            try:
                inputs_x, inputs_x2, labels_x, w_x = next(labeled_train_iter)
            except:
                labeled_train_iter = iter(labeled_loader)
                continue

            try:
                inputs_u, inputs_u2 = next(unlabeled_train_iter)
            except:
                unlabeled_train_iter = iter(unlabeled_loader)
                inputs_u, inputs_u2 = next(unlabeled_train_iter)               
            batch_size = inputs_x.size(0)
               
            # Transform label to one-hot
            raw_labels_x = deepcopy(labels_x)
            labels_x = torch.zeros(batch_size, args.n_cls).scatter_(1, labels_x.view(-1,1), 1)     
            w_x = w_x.view(-1,1).type(torch.FloatTensor) 

            inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
            inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

            with torch.no_grad():
                # label co-guessing of unlabeled samples
                outputs_u11 = classifier(net(inputs_u))
                outputs_u12 = classifier(net(inputs_u2))

                outputs_u21 = classifier2(net2(inputs_u))
                outputs_u22 = classifier2(net2(inputs_u2))           
                
                pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
                ptu = pu**(1 / args.T) # temparature sharpening
                
                targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
                targets_u = targets_u.detach()       
                
                # label co-refinement of labeled samples
                outputs_x = classifier(net(inputs_x))
                outputs_x2 = classifier(net(inputs_x2))         
                
                px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
                px = w_x * labels_x + (1 - w_x) * px              
                ptx = px**(1 / args.T) # temparature sharpening 
                        
                targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
                targets_x = targets_x.detach()       
            
            # mixmatch
            l = np.random.beta(args.alpha, args.alpha)        
            l = max(l, 1-l)
                    
            all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            
            mixed_input = l * input_a + (1 - l) * input_b        
            mixed_target = l * target_a + (1 - l) * target_b
                    
            logits = classifier(net(mixed_input))
            logits_x = logits[:batch_size*2]
            logits_u = logits[batch_size*2:]        
            
            Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch + batch_idx / num_iter, args.warmup)
            
            # regularization
            prior = torch.ones(args.n_cls) / args.n_cls
            prior = prior.cuda()        
            pred_mean = torch.softmax(logits, dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean))

            loss = Lx + lamb * Lu  + penalty

            losses.update(loss.item(), batch_size)
            [acc1, acc5], _ = accuracy(logits_x[:batch_size], raw_labels_x.cuda(), topk=(1, 5))
            top1.update(acc1[0], batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            scaler.scale(loss).backward()            
            scaler.step(optimizer)
            scaler.update()

            # print info
            if (batch_idx + 1) % args.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, batch_idx + 1, len(labeled_loader), 
                    loss=losses, top1=top1))
                sys.stdout.flush()

        return losses, top1

    gmm_loader = torch.utils.data.DataLoader(
            IndexDataset(merge_dataset), batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)

    clean_prob1 = criterion.fit_gmm(gmm_loader, model1, model_idx=0)
    clean_prob2 = criterion.fit_gmm(gmm_loader, model2, model_idx=1)

    pred1 = (clean_prob1 > args.p_threshold)      
    pred2 = (clean_prob2 > args.p_threshold)

    print('Train Model_1')
    labeled_loader, unlabeled_loader, train_bool = co_divide(train_dataset1, train_dataset2, pred2, clean_prob2)
    if train_bool:
        losses, top1 = mixmatch_train(labeled_loader, unlabeled_loader, model1, model2, classifier1, classifier2, criterion, optimizer1, epoch, args)

    print('Train Model_2')
    labeled_loader, unlabeled_loader, train_bool = co_divide(train_dataset1, train_dataset2, pred1, clean_prob1)
    if train_bool:
        losses, top1 = mixmatch_train(labeled_loader, unlabeled_loader, model2, model1, classifier2, classifier1, criterion, optimizer2, epoch, args)

    del labeled_loader, unlabeled_loader
    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, args, best_acc, best_model):
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
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
    if args.noise_method in ['co_teaching', 'dividemix']:
        model2, classifier2, _, optimizer2 = set_model(args)
    val_criterion = nn.CrossEntropyLoss()
    train_loader, val_loader, merge_dataset, train_dataset1, train_dataset2 = set_loader(args)
    scaler = torch.cuda.amp.GradScaler()
    
    if args.noise_method == 'co_teaching':
        # define drop rate schedule
        rate_schedule = np.ones(args.epochs) * args.forget_rate
        rate_schedule[:args.num_gradual] = np.linspace(0, args.forget_rate**args.exponent, args.num_gradual)

    for epoch in range(1, args.epochs+1):
        adjust_lr_wd(args, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        if args.noise_method == 'co_teaching':
            loss, acc = train_coteaching(train_loader, model, model2, classifier, classifier2, criterion, optimizer, optimizer2, epoch, rate_schedule, scaler, args)
        elif args.noise_method == 'dividemix':
            if epoch <= args.warmup:
                print('Warmup model_1 for DivideMix')
                loss, acc = train_dividemix_warmup(train_loader, model, classifier, nn.CrossEntropyLoss().cuda(), optimizer, epoch, scaler, args)
                print('Warmup model_2 for DivideMix')
                loss, acc = train_dividemix_warmup(train_loader, model2, classifier2, nn.CrossEntropyLoss().cuda(), optimizer2, epoch, scaler, args)
            else:
                if epoch + 1 == args.warmup: del train_loader
                loss, acc = train_dividemix(merge_dataset, train_dataset1, train_dataset2, model, model2, classifier, classifier2, criterion, optimizer, optimizer2, epoch, scaler, args)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2-time1, acc))
        best_acc[2] = acc.item()
        
        # eval for one epoch
        best_acc, best_model = validate(val_loader, model, classifier, val_criterion, args, best_acc, best_model)

        if epoch % args.save_freq == 0:
            save_file = os.path.join(args.save_folder, 'epoch_{}.pth'.format(epoch))
            save_model(model, optimizer, args, epoch, save_file)

    save_file = os.path.join(args.save_folder, 'best.pth')
    model.load_state_dict(best_model)
    save_model(model, optimizer, args, epoch, save_file)
    update_json('%s' % args.model_name, best_acc, path=os.path.join(args.save_dir, 'results.json'))


if __name__ == '__main__':
    main()
