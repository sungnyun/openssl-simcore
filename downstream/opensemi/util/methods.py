import numpy as np
from collections import Counter
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss

def ova_loss(logits_open, label):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    label_s_sp = torch.zeros((logits_open.size(0),
                              logits_open.size(2))).long().to(label.device)
    label_range = torch.range(0, logits_open.size(0) - 1).long()
    label_s_sp[label_range, label] = 1
    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :]
                                                + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(logits_open[:, 0, :]
                                                    + 1e-8) * label_sp_neg, 1)[0])
    Lo = open_loss_neg + open_loss
    return Lo

def ova_ent(logits_open):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    Le = torch.mean(torch.mean(torch.sum(-logits_open *
                                   torch.log(logits_open + 1e-8), 1), 1))
    return Le


class SelfTraining(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.T = args.T
        self.lambda_u = args.lambda_u

    def forward(self, labeled_data, unlabeled_data, model=None, classifier=None, epoch_info=None, teacher_model=None):
        inputs_x, targets_x = labeled_data
        inputs_u, _ = unlabeled_data

        t_model, t_classifier = teacher_model
        
        bsz = inputs_x.shape[0]
        inputs_x, inputs_u = inputs_x.cuda(non_blocking=True), inputs_u.cuda(non_blocking=True)
        targets_x = targets_x.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            logits_x = classifier(model(inputs_x))
            logits_u = classifier(model(inputs_u))
            with torch.no_grad():
                t_logits_u = t_classifier(t_model(inputs_u))

            # calculate losses
            Lx = (1 - self.lambda_u) * ce_loss(logits_x, targets_x, reduction='mean')
            pseudo_label = torch.softmax(t_logits_u / self.T, dim=-1)
            Lu = self.lambda_u * ce_loss(logits_u / self.T, pseudo_label, use_hard_labels=False, reduction='mean').mean()
            loss = Lx + Lu
            
        return loss, Lx, Lu, logits_x, targets_x


class OpenMatch(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.T = args.T
        self.start_fix = args.start_fix
        self.lambda_oem, self.lambda_socr = args.lambda_oem, args.lambda_socr
        self.threshold = args.threshold

    def forward(self, labeled_data, unlabeled_data, unlabeled_data_all, model, classifier, classifier_open, epoch_info):
        (_, inputs_x_s, inputs_x), targets_x = labeled_data
        (inputs_u_w, inputs_u_s, _), _ = unlabeled_data
        (inputs_all_w, inputs_all_s, _), _ = unlabeled_data_all

        b_size = inputs_x.shape[0]
        inputs_all = torch.cat([inputs_all_w, inputs_all_s], 0)
        inputs = torch.cat([inputs_x, inputs_x_s, inputs_all], 0).cuda(non_blocking=True)
        targets_x = targets_x.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            feat = model(inputs)
            logits, logits_open = classifier(feat), classifier_open(feat)
            logits_open_u1, logits_open_u2 = logits_open[2*b_size:].chunk(2)

        Lx = ce_loss(logits[:2*b_size], targets_x.repeat(2), reduction='mean')
        Lo = ova_loss(logits_open[:2*b_size], targets_x.repeat(2))

        L_oem = ova_ent(logits_open_u1) / 2.
        L_oem += ova_ent(logits_open_u2) / 2.

        logits_open_u1 = logits_open_u1.view(logits_open_u1.size(0), 2, -1)
        logits_open_u2 = logits_open_u2.view(logits_open_u2.size(0), 2, -1)
        logits_open_u1 = F.softmax(logits_open_u1, 1)
        logits_open_u2 = F.softmax(logits_open_u2, 1)
        L_socr = torch.mean(torch.sum(torch.sum(torch.abs(logits_open_u1 - logits_open_u2)**2, 1), 1))

        if epoch_info[0] >= self.start_fix:
            inputs_ws = torch.cat([inputs_u_w, inputs_u_s], 0).cuda(non_blocking=True)
            feat_ws = model(inputs_ws)
            logits_fix, logits_open_fix = classifier(feat_ws), classifier_open(feat_ws)
            logits_u_w, logits_u_s = logits_fix.chunk(2)
            pseudo_label = torch.softmax(logits_u_w.detach() / self.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(self.threshold).float()
            L_fix = (F.cross_entropy(logits_u_s,
                                     targets_u,
                                     reduction='none') * mask).mean()
        else:
            L_fix = torch.zeros(1).cuda(non_blocking=True).mean()

        loss = Lx + Lo + self.lambda_oem * L_oem \
                + self.lambda_socr * L_socr + L_fix

        return loss, Lx, Lo, L_oem, L_socr, L_fix, logits[:b_size], targets_x

_method_class_map = {
    'self_training': SelfTraining,
    'openmatch': OpenMatch,
}

def get_semisup_method_class(key):
    if key in _method_class_map:
        return _method_class_map[key]
    else:
        raise ValueError('Invalid method: {}'.format(key))
