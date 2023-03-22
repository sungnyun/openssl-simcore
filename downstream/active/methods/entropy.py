import random
import numpy as np

import torch

from .strategy import Strategy


class EntropySampling(Strategy):
    def query(self, unlabel_idxs, n_query):
        unlabel_idxs = np.array(unlabel_idxs)
        probs = self.predict_prob(unlabel_idxs)

        log_probs = torch.log(probs)

        log_probs[log_probs == float("-inf")] = 0
        log_probs[log_probs == float("inf")] = 0

        U = (probs*log_probs).sum(1)
        return list(unlabel_idxs[U.sort()[1][:n_query]])