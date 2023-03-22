from .random import RandomSampling
from .entropy import EntropySampling
from .coreset import CoreSet
from .badge import BadgeSampling


def query_samples(args, dataset, backbone, classifier, unlabel_idxs, label_idxs=None):
    if label_idxs is None:
        strategy = RandomSampling(args, dataset, backbone, classifier)
        selected_indices = strategy.query(unlabel_idxs, args.n_query)
    else:
        if  args.active_method == 'random':
            strategy = RandomSampling(args, dataset, backbone, classifier)
            selected_indices = strategy.query(unlabel_idxs, args.n_query)
        elif args.active_method == 'entropy':
            strategy = EntropySampling(args, dataset, backbone, classifier)
            selected_indices = strategy.query(unlabel_idxs, args.n_query)
        elif args.active_method == 'coreset':
            strategy = CoreSet(args, dataset, backbone, classifier)
            selected_indices = strategy.query(label_idxs, unlabel_idxs, args.n_query)
        elif args.active_method == 'badge':
            strategy = BadgeSampling(args, dataset, backbone, classifier)
            selected_indices = strategy.query(unlabel_idxs, args.n_query)

        selected_indices.extend(label_idxs)

    return selected_indices