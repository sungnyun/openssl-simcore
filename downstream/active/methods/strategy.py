import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label, item


class Strategy:
    def __init__(self, args, dataset, backbone, classifier):
        self.args = args
        self.dataset = dataset
        self.backbone = backbone
        self.classifier = classifier
        self.loss_func = nn.CrossEntropyLoss()
        
    def query(self, label_idxs, unlabel_idx, n_query):
        pass
    
    # Entropy Sampling
    def predict_prob(self, unlabel_idxs):
        loader_te = DataLoader(DatasetSplit(self.dataset, unlabel_idxs), shuffle=False)
        
        self.backbone.eval()
        probs = torch.zeros([len(unlabel_idxs), self.args.n_cls])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda(non_blocking=True)), Variable(y.cuda(non_blocking=True))
                output = self.classifier(self.backbone(x))
                probs[idxs] = torch.nn.functional.softmax(output, dim=1).cpu().data
        return probs

    # CoreSet
    def get_embedding(self, data_idxs):
        loader_te = DataLoader(DatasetSplit(self.dataset, data_idxs), shuffle=False)
        
        self.backbone.eval()
        embedding = torch.zeros([len(data_idxs), self.backbone.module.final_feat_dim])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda(non_blocking=True)), Variable(y.cuda(non_blocking=True))
                emb = self.backbone(x)
                embedding[idxs] = emb.cpu().data
        
        return embedding
        
    # BADGE: gradient embedding (assumes cross-entropy loss)
    def get_grad_embedding(self, data_idxs):
        embDim = self.backbone.module.final_feat_dim
        self.backbone.eval()
        
        nLab = self.args.n_cls 
        embedding = np.zeros([len(data_idxs), embDim * nLab])
        loader_te = DataLoader(DatasetSplit(self.dataset, data_idxs), shuffle=False)
        
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda(non_blocking=True)), Variable(y.cuda(non_blocking=True))
                out = self.backbone(x)
                cout = self.classifier(out)

                out = out.data.cpu().numpy()
                
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)
                
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)