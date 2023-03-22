import numpy as np
from sklearn.mixture import GaussianMixture

import torch 
import torch.nn as nn
import torch.nn.functional as F


# Loss functions
class CoTeaching(nn.Module):
    def forward(self, y_1, y_2, t, forget_rate):
        loss_1 = F.cross_entropy(y_1, t, reduce = False)
        ind_1_sorted = torch.argsort(loss_1.data)
        loss_1_sorted = loss_1[ind_1_sorted]

        loss_2 = F.cross_entropy(y_2, t, reduce = False)
        ind_2_sorted = torch.argsort(loss_2.data)
        loss_2_sorted = loss_2[ind_2_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_1_sorted))

        ind_1_update=ind_1_sorted[:num_remember]
        ind_2_update=ind_2_sorted[:num_remember]
        # exchange
        loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
        loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

        return loss_1_update, loss_2_update


class DivideMix(nn.Module):
    """ we assume that WebFG dataset has moderate noise_ratio, based on SimCore sampling ratio, and assym noise_mode.
    """
    def __init__(self, lambda_u):
        super(DivideMix, self).__init__()

        self.all_loss = [[],[]] # save the history of losses from two networks
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.lambda_u = lambda_u

    def fit_gmm(self, train_loader, model, model_idx):    
        model.eval()

        losses = torch.zeros(len(train_loader.dataset))    
        with torch.no_grad():
            for batch_idx, (inputs, targets, index) in enumerate(train_loader):
                inputs, targets = inputs.cuda(), targets.cuda() 
                outputs = model(inputs) 
                loss = self.ce(outputs, targets)  
                for b in range(inputs.size(0)):
                    losses[index[b]] = loss[b]         
        losses = (losses - losses.min()) / (losses.max() - losses.min())    
        self.all_loss[model_idx].append(losses)

        history = torch.stack(self.all_loss[model_idx])
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
        # input_loss = losses.reshape(-1,1)
        
        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss) 
        prob = prob[:, gmm.means_.argmin()]       

        return prob

    def linear_rampup(self, current, warm_up, rampup_length=16):
        current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
        return self.lambda_u * float(current)

    def forward(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, self.linear_rampup(epoch, warm_up)