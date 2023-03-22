import copy
import numpy as np
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F 

from ssl.base import BaseSelfSupervisedModel


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=4096, out_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        
        return x 


@torch.no_grad()
def distributed_sinkhorn(out, epsilon):
    epsilon = epsilon
    sinkhorn_iterations = 3

    Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] # * args.world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    # dist.all_reduce(sum_Q)
    Q /= sum_Q
    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        # dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out


class SwAV(BaseSelfSupervisedModel):
    def __init__(self, backbone: nn.Module, params: Namespace):
        super().__init__(backbone, params)
        
        self.online_encoder = self.backbone
        self.online_projector = projection_MLP(self.backbone.final_feat_dim)
    
        # prototype layer
        nmb_prototypes = params.prototypes

        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(self.online_projector.out_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(self.online_projector.out_dim, nmb_prototypes, bias=False)

        # build the queue
        self.queue = None
        self.queue_length = 0
        self.queue_length -= self.queue_length % params.batch_size
        self.epoch_queue_starts = 15
        self.crops_for_assign = [0, 1]
        self.nmb_crops = [2] if not params.local_crops_number else [2, params.local_crops_number]
        self.bs = params.batch_size
        self.T = 0.1
        self.epsilon = 0.02  # default is 0.05 but try lower value when getting stuck at 8.006

    def _data_parallel(self):
        self.online_encoder = nn.DataParallel(self.online_encoder)

    def compute_ssl_loss(self, x1, x2=None, return_features=False):
        device = x1.device

        if not isinstance(x2, list): 
            x = torch.cat([x1, x2], dim=0)
        else:  # use multi-crop strategy
            x = [x1, torch.cat(x2, dim=0)]

        # optionally starts a queue
        if self.queue_length > 0 and self.epoch >= self.epoch_queue_starts and self.queue is None:
            self.queue = torch.zeros(
                len(self.crops_for_assign),
                self.queue_length,
                self.online_projector.out_dim,
            ).to(device)

        # normalize the prototypes
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.prototypes.weight.copy_(w)

        # ============ multi-res forward passes ... ============
        if not isinstance(x, list):
            embedding = self.online_projector(self.online_encoder(x))
        else:  # use multi-crop strategy
            embedding = torch.empty(0).to(device)
            for inp in x:
                _emb = self.online_projector(self.online_encoder(inp))
                embedding = torch.cat([embedding, _emb])

        # normalize
        embedding = F.normalize(embedding, dim=1)
        
        output = self.prototypes(embedding)
        embedding = embedding.detach()
        bs = self.bs

        # ============ swav loss ... ============
        loss = torch.tensor(0.).to(device)
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                # time to use the queue
                if self.queue is not None:
                    use_the_queue = True
                    out = torch.cat((torch.mm(
                        self.queue[i],
                        self.prototypes.weight.t()
                    ), out))

                    # fill the queue
                    self.queue[i, bs:] = self.queue[i, :-bs].clone()
                    self.queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                # get assignments
                q = distributed_sinkhorn(out, self.epsilon)[-bs:]

            # cluster assignment prediction
            subloss = torch.tensor(0.).to(device)
            for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                x = output[bs * v: bs * (v + 1)] / self.T
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            loss += subloss / (sum(self.nmb_crops) - 1)
        loss /= len(self.crops_for_assign)
    
        if return_features:
            return loss, embedding[:self.bs], embedding[self.bs:]
        else:
            return loss

    def on_epoch_start(self, epoch):
        self.epoch = epoch