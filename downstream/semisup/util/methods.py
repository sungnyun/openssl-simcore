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


def linear_rampup(current, tot_epochs):
    current = np.clip(current / tot_epochs, 0.0, 1.0)
    return float(current)


class MixMatch(nn.Module):
    """ refer to https://github.com/YU1ut/MixMatch-pytorch/blob/master/train.py
    """
    def __init__(self, args):
        super().__init__()
        
        self.n_cls = args.n_cls
        self.T = args.T
        self.mixup_beta = args.mixup_beta
        self.tot_epochs = args.epochs
        self.eval_step = args.eval_step
        self.lambda_u = args.lambda_u

    def one_hot(self, targets, nClass):
        logits = torch.zeros(targets.size(0), nClass).cuda(non_blocking=True)
        return logits.scatter_(1,targets.unsqueeze(1),1)

    def _interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self._interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    def forward(self, labeled_data, unlabeled_data, model=None, classifier=None, epoch_info=None):
        inputs_x, targets_x = labeled_data
        _, (inputs_u, _) = unlabeled_data
        inputs_u_1, inputs_u_2 = inputs_u
        
        batch_size = inputs_x.size(0)

        inputs_x, targets_x = inputs_x.cuda(non_blocking=True), targets_x.cuda(non_blocking=True)
        inputs_u_1, inputs_u_2 = inputs_u_1.cuda(non_blocking=True), inputs_u_2.cuda(non_blocking=True)
        
        with torch.cuda.amp.autocast():
            # Transform label to one-hot
            targets_l = self.one_hot(targets_x, self.n_cls)
            
            with torch.no_grad():
                # compute guessed labels of unlabel samples
                outputs_u_1 = classifier(model(inputs_u_1))
                outputs_u_2 = classifier(model(inputs_u_2))
                p = (torch.softmax(outputs_u_1, dim=1) + torch.softmax(outputs_u_2, dim=1)) / 2
                pt = p**(1/self.T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()

            # mixup
            all_inputs = torch.cat([inputs_x, inputs_u_1, inputs_u_2], dim=0)
            all_targets = torch.cat([targets_l, targets_u, targets_u], dim=0)

            l = np.random.beta(self.mixup_beta, self.mixup_beta)
            l = max(l, 1-l)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
            mixed_input = list(torch.split(mixed_input, batch_size))
            mixed_input = self.interleave(mixed_input, batch_size)

            logits = [classifier(model(mixed_input[0]))]
            for input in mixed_input[1:]:
                logits.append(classifier(model(input)))

            # put interleaved samples back
            logits = self.interleave(logits, batch_size)
            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)

            probs_u = torch.softmax(logits_u, dim=1)

            Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * mixed_target[:batch_size], dim=1))
            Lu = torch.mean((probs_u - mixed_target[batch_size:])**2)
            Lu *= (self.lambda_u * linear_rampup(epoch_info[0] + epoch_info[1]/self.eval_step, self.tot_epochs))

            loss = Lx + Lu
        
        return loss, Lx, Lu, logits_x, targets_x


class ReMixMatch(nn.Module):
    """ refer to https://github.com/TorchSSL/TorchSSL/blob/4856f9cc03b762e5549719541b19d6da4c7438a7/models/remixmatch/remixmatch.py
    """
    def __init__(self, args):
        super().__init__()
        
        self.T = args.T
        self.n_cls = args.n_cls
        self.mixup_beta = args.mixup_beta
        self.tot_epochs = args.epochs
        self.eval_step = args.eval_step
        self.p_target = args.p_target
        self.p_model = None

        self.w_rot = 0.5
        self.w_match = 1.5
        self.w_kl = 0.5

    def one_hot(self, targets, nClass):
        logits = torch.zeros(targets.size(0), nClass).cuda(non_blocking=True)
        return logits.scatter_(1,targets.unsqueeze(1),1)

    def mixup_one_target(self, x, y, alpha=1.0, is_bias=False):
        """Returns mixed inputs, mixed targets, and lambda
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        if is_bias: lam = max(lam, 1-lam)

        index = torch.randperm(x.size(0)).cuda(non_blocking=True)

        mixed_x = lam*x + (1-lam)*x[index, :]
        mixed_y = lam*y + (1-lam)*y[index]
        return mixed_x, mixed_y, lam

    def _interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self._interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    def forward(self, labeled_data, unlabeled_data, model=None, classifier=None, epoch_info=None):
        inputs_x, targets_x = labeled_data
        _, (inputs_u, _) = unlabeled_data
        inputs_u_w, inputs_u_s1, inputs_u_s2, inputs_u_rot, rot_v = inputs_u
        batch_size = inputs_x.size(0)

        inputs_x, targets_x = inputs_x.cuda(non_blocking=True), targets_x.cuda(non_blocking=True)
        inputs_u_w, inputs_u_s1, inputs_u_s2  = inputs_u_w.cuda(non_blocking=True), inputs_u_s1.cuda(non_blocking=True), inputs_u_s2.cuda(non_blocking=True)
        # we exclude rotation loss
        # inputs_u_rot = inputs_u_rot.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                outputs_u_w = classifier(model(inputs_u_w))

                prob_u_w = torch.softmax(outputs_u_w, dim=1)

                if self.p_model == None:
                    self.p_model = torch.mean(prob_u_w.detach(), dim=0)
                else:
                    self.p_model = self.p_model * 0.999 + torch.mean(prob_u_w.detach(), dim=0) * 0.001

                prob_u_w = prob_u_w * torch.tensor(self.p_target).cuda() / self.p_model
                prob_u_w = (prob_u_w / prob_u_w.sum(dim=-1, keepdim=True))

                sharpen_prob_u_w = prob_u_w ** (1 / self.T)
                sharpen_prob_u_w = (sharpen_prob_u_w / sharpen_prob_u_w.sum(dim=-1, keepdim=True)).detach()

                # MixUp
                mixed_inputs = torch.cat((inputs_x, inputs_u_s1, inputs_u_s2, inputs_u_w))
                input_labels = torch.cat([self.one_hot(targets_x, self.n_cls), sharpen_prob_u_w, sharpen_prob_u_w, sharpen_prob_u_w], dim=0)

                mixed_x, mixed_y, _ = self.mixup_one_target(mixed_inputs, 
                                                    input_labels,
                                                    self.mixup_beta,
                                                    is_bias=True)

                # Interleave labeled and unlabeled samples between batches to get correct batch norm calculation
                mixed_x = list(torch.split(mixed_x, batch_size))
                mixed_x = self.interleave(mixed_x, batch_size)  

            logits = [classifier(model(mixed_x[0]))]
            for ipt in mixed_x[1:]:
                logits.append(classifier(model(ipt)))

            u1_logits = classifier(model(inputs_u_s1))
            # logits_rot = classifier_rot(model(inputs_u_rot)[1])  # need an auxiliary classifier that has 4 classes 
            logits = self.interleave(logits, batch_size)
            model.train()

            logits_x = logits[0]
            logits_u = torch.cat(logits[1:])

            # calculate rot loss with w_rot
            # rot_loss = ce_loss(logits_rot, rot_v, reduction='mean')
            # rot_loss = rot_loss.mean()

            # sup loss
            Lx = ce_loss(logits_x, mixed_y[:batch_size], use_hard_labels=False)
            Lx = Lx.mean()

            # unsup_loss
            Lu = ce_loss(logits_u, mixed_y[batch_size:], use_hard_labels=False)
            Lu = Lu.mean()

            # loss U1
            u1_loss = ce_loss(u1_logits, sharpen_prob_u_w, use_hard_labels=False)
            u1_loss = u1_loss.mean()

            # ramp for w_match
            w_match = self.w_match * linear_rampup(epoch_info[0] + epoch_info[1]/self.eval_step, self.tot_epochs)
            w_kl = self.w_kl * linear_rampup(epoch_info[0] + epoch_info[1]/self.eval_step, self.tot_epochs)
            
            Lu *= w_match
            loss = Lx + Lu + w_kl * u1_loss  # + self.w_rot * rot_loss
        
        return loss, Lx, Lu, logits_x, targets_x


class FixMatch(nn.Module):
    """ refer to https://github.com/kekmodel/FixMatch-pytorch/blob/master/train.py
    """
    def __init__(self, args):
        super().__init__()
        
        self.mu = args.mu
        self.hard_label = args.hard_label
        self.T = args.T
        self.threshold = args.threshold
        self.lambda_u = args.lambda_u

    def consistency_loss(self, logits_s, logits_w, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True):
        assert name in ['ce', 'L2']
        logits_w = logits_w.detach()
        if name == 'L2':
            assert logits_w.size() == logits_s.size()
            return F.mse_loss(logits_s, logits_w, reduction='mean')

        elif name == 'L2_mask':
            pass

        elif name == 'ce':
            pseudo_label = torch.softmax(logits_w, dim=-1)
            max_probs, max_idx = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(p_cutoff).float()
            select = max_probs.ge(p_cutoff).long()
            # strong_prob, strong_idx = torch.max(torch.softmax(logits_s, dim=-1), dim=-1)
            # strong_select = strong_prob.ge(p_cutoff).long()
            # select = select * strong_select * (strong_idx == max_idx)
            if use_hard_labels:
                masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
            else:
                pseudo_label = torch.softmax(logits_w / T, dim=-1)
                masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
            return masked_loss.mean(), mask.mean(), select, max_idx.long()

        else:
            assert Exception('Not Implemented consistency_loss')

    def forward(self, labeled_data, unlabeled_data, model=None, classifier=None, epoch_info=None):
        inputs_x, targets_x = labeled_data
        _, (inputs_u, _) = unlabeled_data
        inputs_u_w, inputs_u_s = inputs_u
        
        bsz = inputs_x.shape[0]
        inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s)).cuda(non_blocking=True)
        targets_x = targets_x.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            output = classifier(model(inputs))

            logits_x = output[:bsz]
            logits_u_w, logits_u_s = output[bsz:].chunk(2)
            del output

            # calculate losses
            Lx = ce_loss(logits_x, targets_x, reduction='mean')

            Lu, mask, select, pseudo_lb = self.consistency_loss(logits_u_s, logits_u_w, 'ce', self.T, self.threshold, use_hard_labels=self.hard_label)
            Lu *= self.lambda_u

            loss = Lx + Lu
            
        return loss, Lx, Lu, logits_x, targets_x


class FlexMatch(nn.Module):
    """ refer to https://github.com/TorchSSL/TorchSSL/blob/4856f9cc03b762e5549719541b19d6da4c7438a7/models/flexmatch/flexmatch.py
    """
    def __init__(self, args):
        super().__init__()
        
        self.p_target = args.p_target
        self.p_model = None

        self.selected_label = args.selected_label
        self.classwise_acc = args.classwise_acc

        self.n_cls = args.n_cls
        self.thresh_warmup = args.thresh_warmup
        self.T = args.T
        self.p_cutoff = args.p_cutoff
        self.lambda_u = args.lambda_u
        self.hard_label = args.hard_label
        self.use_DA = args.use_DA

    def consistency_loss(self, logits_s, logits_w, class_acc, name='ce',
                        T=1.0, p_cutoff=0.0, use_hard_labels=True, use_DA=False):
        assert name in ['ce', 'L2']
        logits_w = logits_w.detach()
        if name == 'L2':
            assert logits_w.size() == logits_s.size()
            return F.mse_loss(logits_s, logits_w, reduction='mean')

        elif name == 'L2_mask':
            pass

        elif name == 'ce':
            pseudo_label = torch.softmax(logits_w, dim=-1)
            if use_DA:
                if self.p_model == None:
                    self.p_model = torch.mean(pseudo_label.detach(), dim=0)
                else:
                    self.p_model = self.p_model * 0.999 + torch.mean(pseudo_label.detach(), dim=0) * 0.001
                pseudo_label = pseudo_label * torch.tensor(self.p_target).cuda() / self.p_model
                pseudo_label = (pseudo_label / pseudo_label.sum(dim=-1, keepdim=True))

            max_probs, max_idx = torch.max(pseudo_label, dim=-1)
            # mask = max_probs.ge(p_cutoff * (class_acc[max_idx] + 1.) / 2).float()  # linear
            # mask = max_probs.ge(p_cutoff * (1 / (2. - class_acc[max_idx]))).float()  # low_limit
            mask = max_probs.ge(p_cutoff * (class_acc[max_idx] / (2. - class_acc[max_idx]))).float()  # convex
            # mask = max_probs.ge(p_cutoff * (torch.log(class_acc[max_idx] + 1.) + 0.5)/(math.log(2) + 0.5)).float()  # concave
            select = max_probs.ge(p_cutoff).long()
            if use_hard_labels:
                masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
            else:
                pseudo_label = torch.softmax(logits_w / T, dim=-1)
                masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
            return masked_loss.mean(), mask.mean(), select, max_idx.long()

        else:
            assert Exception('Not Implemented consistency_loss')

    def forward(self, labeled_data, unlabeled_data, model=None, classifier=None, epoch_info=None):
        inputs_x, targets_x = labeled_data
        inputs_u_idx, (inputs_u, _) = unlabeled_data
        inputs_u_w, inputs_u_s = inputs_u
        
        bsz = inputs_x.shape[0]

        inputs_x, inputs_u_w, inputs_u_s = inputs_x.cuda(non_blocking=True), inputs_u_w.cuda(non_blocking=True), inputs_u_s.cuda(non_blocking=True)
        inputs_u_idx = inputs_u_idx.cuda(non_blocking=True)
        targets_x = targets_x.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            pseudo_counter = Counter(self.selected_label.tolist())
            if max(pseudo_counter.values()) < len(self.selected_label):  # not all(5w) -1
                if self.thresh_warmup:
                    for i in range(self.n_cls):
                        self.classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
                else:
                    wo_negative_one = deepcopy(pseudo_counter)
                    if -1 in wo_negative_one.keys():
                        wo_negative_one.pop(-1)
                    for i in range(self.n_cls):
                        self.classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())

            inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s))

            logits = classifier(model(inputs))
            logits_x = logits[:bsz]
            logits_u_w, logits_u_s = logits[bsz:].chunk(2)
            del logits

            # calculate losses
            Lx = ce_loss(logits_x, targets_x, reduction='mean')

            Lu, mask, select, pseudo_lb = self.consistency_loss(logits_u_s,
                                                                logits_u_w,
                                                                self.classwise_acc,
                                                                'ce', self.T, self.p_cutoff,
                                                                use_hard_labels=self.hard_label,
                                                                use_DA=self.use_DA)

            if inputs_u_idx[select == 1].nelement() != 0:
                self.selected_label[inputs_u_idx[select == 1]] = pseudo_lb[select == 1]

            Lu *= self.lambda_u
            loss = Lx + Lu

        return loss, Lx, Lu, logits_x, targets_x

_method_class_map = {
    'mixmatch': MixMatch,
    'remixmatch': ReMixMatch,
    'fixmatch': FixMatch,
    'flexmatch': FlexMatch
}

def get_semisup_method_class(key):
    if key in _method_class_map:
        return _method_class_map[key]
    else:
        raise ValueError('Invalid method: {}'.format(key))
