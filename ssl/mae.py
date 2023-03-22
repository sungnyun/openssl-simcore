### refer to https://github.com/facebookresearch/mae ###

import copy
import math
import random
import warnings
from argparse import Namespace
from functools import wraps

import torch
import torch.nn.functional as F
from torch import nn

from ssl.base import BaseSelfSupervisedModel


def _get_module_device(module):
    return next(module.parameters()).device


class MAE(BaseSelfSupervisedModel):
    def __init__(self, backbone: nn.Module, params: Namespace):
        super().__init__(backbone, params)

        self.norm_pix_loss = params.norm_pix_loss

        backbone.set_mask_ratio(mask_ratio=params.mask_ratio)
        self.online_encoder = backbone  # for consistency

        # get device of network and make wrapper same device
        device = _get_module_device(self.online_encoder)
        self.to(device)

    def _data_parallel(self):
        self.online_encoder = nn.DataParallel(self.online_encoder)

    def compute_ssl_loss(self, x, _, return_features=False):
        pred, target, mask = self.online_encoder(x, pretrain=True)  # pred: [N, L, p*p*3]

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        # MSE loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        # mask: [N, L], 0 is keep, 1 is remove
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        return loss

    def forward_features(self, x):
        """ Only used in train_selfsup_sampling.py
        """
        output = self.backbone(x, global_pool=True)
        return output