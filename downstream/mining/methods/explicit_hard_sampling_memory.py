from argparse import Namespace
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .simclr import SimCLR, NTXentLoss


class ExpHardNTXentLoss(NTXentLoss):
    def __init__(self, temperature, use_cosine_similarity, sampling_ratio):
        super(ExpHardNTXentLoss, self).__init__(temperature, use_cosine_similarity)
        self.sampling_ratio = sampling_ratio
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, zis, zjs, queue):
        batch_size = zis.shape[0]
        K = queue.shape[1]

        representations = torch.cat([zjs, zis], dim=0)
        representations = torch.nn.functional.normalize(representations, dim=-1)
        device = representations.device

        similarity_matrix = self.similarity_function(
            representations, representations)

        similarity_matrix_queue = self.similarity_function(
            representations, queue.T) # (2N, K)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        mask = self._get_correlated_mask(batch_size).to(device)
        negatives = similarity_matrix[mask].view(2 * batch_size, -1)

        sampling_num = int(K * self.sampling_ratio)
        negatives_queue = similarity_matrix_queue.sort(descending=True, dim=1)[0][:, :sampling_num]
        neg = torch.cat([negatives, negatives_queue], dim=1) # (2N, 2N-2+sampling_num)
        logits = torch.cat((positives, neg), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * batch_size).to(representations.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * batch_size)


class ExpHardSamplingMemory(SimCLR):
    def __init__(self, backbone: nn.Module, params: Namespace):
        super().__init__(backbone, params)
        simclr_projection_dim = 128
        simclr_temperature = 0.07
        sampling_ratio = params.sampling_ratio
        self.K = 65536
        # self.K = params.K
        
        self.ssl_loss_fn = ExpHardNTXentLoss(temperature=simclr_temperature, use_cosine_similarity=False, 
                                             sampling_ratio=sampling_ratio)

        self.register_buffer("queue", torch.randn(simclr_projection_dim, self.K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def on_step_start(self, train_loader):
        global train_iter
        try:
            images, _ = next(train_iter)
        except:
            print('Iterator restart!')
            train_iter = iter(train_loader)
            images, _ = next(train_iter)
        img = images.cuda(non_blocking=True)
        
        with torch.cuda.amp.autocast():
            p1 = self.head(self.backbone(img))
            p1 = F.normalize(p1, dim=1)
            self._dequeue_and_enqueue(p1.clone().detach())

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
        loss = self.ssl_loss_fn(p1, p2, self.queue.clone().detach())

        if return_features:
            if x2 is None:
                return loss, f
            else:
                return loss, f1, f2
        else:
            return loss
