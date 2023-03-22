from argparse import Namespace
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .simclr import SimCLR, NTXentLoss


class HardNTXentLoss(NTXentLoss):
    def __init__(self, temperature, use_cosine_similarity, beta, tau_plus):
        super(HardNTXentLoss, self).__init__(temperature, use_cosine_similarity)
        self.beta = beta
        self.tau_plus = tau_plus

    def forward(self, zis, zjs):
        batch_size = zis.shape[0]
        representations = torch.cat([zjs, zis], dim=0)
        representations = torch.nn.functional.normalize(representations, dim=-1)
        device = representations.device

        similarity_matrix = self.similarity_function(
            representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        mask = self._get_correlated_mask(batch_size).to(device)
        negatives = similarity_matrix[mask].view(2 * batch_size, -1)

        neg = (negatives / self.temperature).exp() # (2N, 2N-2)
        pos = (positives / self.temperature).exp()

        N = batch_size * 2 - 2
        imp = (self.beta* neg.log()).exp()
        reweight_neg = (imp*neg).sum(dim = -1) / imp.mean(dim = -1)
        Ng = (-self.tau_plus * N * pos + reweight_neg) / (1 - self.tau_plus)
        # constrain (optional)
        Ng = torch.clamp(Ng, min = N * np.e**(-1 / self.temperature))

        loss = (- torch.log(pos / (pos + Ng) )).mean()

        return loss


class HardSampling(SimCLR):
    def __init__(self, backbone: nn.Module, params: Namespace):
        super().__init__(backbone, params)
        simclr_projection_dim = 128
        simclr_temperature = 0.07
        beta = params.beta
        tau_plus = params.tau_plus

        self.ssl_loss_fn = HardNTXentLoss(temperature=simclr_temperature, use_cosine_similarity=False, 
                                          beta=beta, tau_plus=tau_plus)

    @torch.no_grad()
    def on_step_start(self, train_loader):
        pass

    def compute_ssl_loss(self, x1, x2=None, return_features=False):
        if x2 is None:
            x = x1
        else:
            x = torch.cat([x1, x2])
        batch_size = int(x.shape[0] / 2)
            
        f = self.backbone(x)
        f1, f2 = f[:batch_size], f[batch_size:]
        p1 = self.head(f1)
        p2 = self.head(f2)
        loss = self.ssl_loss_fn(p1, p2)

        if return_features:
            if x2 is None:
                return loss, f
            else:
                return loss, f1, f2
        else:
            return loss