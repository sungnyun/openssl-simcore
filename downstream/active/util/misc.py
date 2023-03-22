from __future__ import print_function

import os
import json
import math
import copy
import numpy as np

import torch
import torch.optim as optim

__all__ = ['AverageMeter', 'AverageClassMeter', 'adjust_lr_wd', 'warmup_learning_rate', 'accuracy', 'update_metric', 'get_best_acc', 'save_model', 'update_json']


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageClassMeter(object):
    def __init__(self, n_cls):
        self.meter = []
        for _ in range(n_cls): self.meter.append(AverageMeter())
        self.n_cls = n_cls

    def update(self, cls, val, n=1):
        self.meter[cls].update(val, n)

        self.val = self.meter[-1].val
        self.avg = torch.tensor(sum([m.avg for m in self.meter]) / self.n_cls)


def adjust_lr_wd(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    wd = args.weight_decay
    if args.wd_scheduler:
        wd_min = args.weight_decay_end
        wd = wd_min + (wd - wd_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr
        if i == 0: # in case of DINO-ViT and MAE-ViT, only wd for regularized params
            param_group['weight_decay'] = wd


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        n_cls = output.shape[1]
        valid_topk = [k for k in topk if k <= n_cls]
        
        maxk = max(valid_topk)
        bsz = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            if k in valid_topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / bsz))
            else: res.append(torch.tensor([0.]))
        return res, bsz


def update_metric(output, labels, top1, top5, args):
    if top1.__class__.__name__ == 'AverageMeter':
        [acc1, acc5], bsz = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)
        top5.update(acc5[0], bsz)
    else: # mean per-class accuracy
        for cls in range(args.n_cls):
            if not (labels==cls).sum(): continue
            [acc1, acc5], bsz = accuracy(output[labels==cls], labels[labels==cls], topk=(1, 5))
            top1.update(cls, acc1[0], bsz)
            top5.update(cls, acc5[0], bsz)
    return top1, top5


def get_best_acc(val_acc1, val_acc5, best_acc):
    best = False
    if val_acc1.item() > best_acc[0]:
        best_acc[0] = val_acc1.item()
        best_acc[1] = val_acc5.item()
        best = True
    return best_acc, best


def save_model(model, optimizer, args, epoch, save_file, indices=None, classifier=None):
    print('==> Saving...')
    state = {
        'args': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    if indices is not None:
        state['indices'] = indices
    if classifier is not None:  # for active learning fine-tuning and open-set semi
        state['classifier'] = classifier.state_dict()
    torch.save(state, save_file)
    del state
    
    
def update_json(exp_name, acc=[], path='./save/results.json'):
    acc = [round(a, 2) for a in acc]
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump({}, f)

    with open(path, 'r', encoding="UTF-8") as f:
        result_dict = json.load(f)
        result_dict[exp_name] = acc
    
    with open(path, 'w') as f:
        json.dump(result_dict, f)
        
    print('best accuracy: {} (Acc@1, Acc@5, Train Acc)'.format(acc))
    print('results updated to %s' % path)
    