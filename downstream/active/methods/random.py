import random

from .strategy import Strategy


class RandomSampling(Strategy):
    def query(self, unlabel_idxs, n_query):
        return random.sample(unlabel_idxs, n_query)