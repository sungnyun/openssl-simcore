from __future__ import print_function

import os
import math
import numpy as np

import torch
import torch.optim as optim

__all__ = ['AverageMeter', 'adjust_lr_wd', 'warmup_learning_rate', 'save_model']


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
    