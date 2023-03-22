import copy
import random
import numpy as np
from tqdm import tqdm
from collections import deque
from sklearn.cluster import KMeans

import torch
import torch.nn as nn


def random_sampling(model, args):
    from train_selfsup import get_dataset

    # "sample_*" denotes the dataset for "*_sampling" function (val=True)
    sample_dataset2 = get_dataset(args, args.dataset2, args.data_folder2, val=True)
    sampling_nums = int(args.sampling_ratio * len(sample_dataset2))
    selected_indices = random.sample(range(len(sample_dataset2)), sampling_nums)

    print('Complete! {:d} number of indices sampled from {:s}.'.format(len(selected_indices), args.dataset2))

    return list(selected_indices)


# greedy selection algorithm for SimCore algorithm
def greedy(sim, sampling_nums=0):
    N, K = sim.shape
    queue_list = [deque(torch.argsort(sim[:,k], descending=True).numpy()) for k in range(K)]
    indices, tmp = set(), set()
    # for one iteration, each cluster picks one sample with the maximum utility function (i.e., maximum cosine similarity) 
    FLAG, threshold = 1, 0
    while len(indices) < sampling_nums:
        # for-loop according to the centroids
        func_value = 0
        for q, queue in enumerate(queue_list):
            while True:
                i = queue.popleft()
                if i not in indices: break
            tmp.update({i})
            if args.stop:
                func_value += sim[i, q].item() / K

        if FLAG == args.patience:
            threshold = func_value * args.stop_thresh
        FLAG += 1

        if args.stop and func_value < threshold:
            print('Stopped sampling because the rest are not similar to the target dataset.')
            break

        if len(indices) + len(tmp) > sampling_nums:
            print('Stopped sampling because of the limited budget.')
            tmp = set(np.random.choice(list(tmp), size=(sampling_nums - len(indices)), replace=False))
        indices.update(tmp)
        tmp = set()

    return indices


def simcore_sampling(model, args):
    from train_selfsup import get_dataset

    # "sample_*" denotes the dataset for "*_sampling" function (val=True)
    sample_dataset1 = get_dataset(args, args.dataset1, args.data_folder1, val=True)
    sample_dataset2 = get_dataset(args, args.dataset2, args.data_folder2, val=True)

    if args.stop:
        sampling_nums = 50 * len(sample_dataset1)
    else:
        sampling_nums = int(args.sampling_ratio * len(sample_dataset2))

    if sampling_nums == 0:
        assert args.from_ssl_official
        print('not sampling from open-set, and this is only for finetuning on X with a SSL official checkpoint')
        return []

    dataloader1 = torch.utils.data.DataLoader(sample_dataset1, batch_size=args.batch_size, shuffle=False, num_workers=1)
    dataloader2 = torch.utils.data.DataLoader(sample_dataset2, batch_size=args.batch_size, shuffle=False, num_workers=8)

    if args.method == 'mae':
        model.backbone.set_mask_ratio(mask_ratio=0)

    print('SimCore sampling starts!')
    model.eval()    
    with torch.no_grad():
        feats = []
        for images, _ in tqdm(dataloader1):
            img = images.cuda(non_blocking=True)
            
            z = model.forward_features(img)
            feats.append(nn.functional.normalize(z, dim=1).cpu())
        feats = torch.cat(feats, dim=0)

        # k-means clustering on normalized features
        if args.cluster_num < len(sample_dataset1):
            kmeans = KMeans(n_clusters=args.cluster_num, random_state=0).fit(feats.numpy())
            centroids = torch.tensor(kmeans.cluster_centers_).cuda()
        else:
            print('Using the centroids as each datapoint in {}.'.format(args.dataset1))
            centroids = feats.cuda()
        centroids = nn.functional.normalize(centroids, dim=1)

        del dataloader1
        print('Finding the closest {:d} samples from {:s} ...'.format(sampling_nums, args.dataset2))
        # sim is cosine similarity between centroids of target dataset(centroids) and features of openset(z)
        # sim = torch.tensor([], device=torch.device('cpu'))
        sim = []
        for idx, (images, _) in tqdm(enumerate(dataloader2)):
            img = images.cuda(non_blocking=True)
            z = model.forward_features(img)
            z = nn.functional.normalize(z, dim=1)
            
            sim.append(torch.mm(centroids, z.T).cpu())
        sim = torch.cat(sim, dim=1)
        print('Cosine similarity matrix is computed...')
        
        # get the solution of facility location problem in an iterative fashion
        selected_indices = greedy(sim.T.cpu(), sampling_nums=sampling_nums) # sim.shape == (# of openset, # of centroids)
        print('Complete! {:d} number of indices sampled from {:s}.'.format(len(selected_indices), args.dataset2))
        del dataloader2
        
    if args.method == 'mae':
        model.backbone.set_mask_ratio(mask_ratio=args.mask_ratio)
    return list(selected_indices)


def get_selected_indices(model, args):
    init_ckpt = copy.deepcopy(model.state_dict())
    if args.sampling_method == 'random':
        pass
    elif args.retrieval_ckpt is not None:
        print('pretrained retrieval model loaded from: {}'.format(args.retrieval_ckpt))
        model.load_state_dict(torch.load(args.retrieval_ckpt)['model'])
    elif args.from_ssl_official:
        print('pretrained retrieval model loaded from SimCLR ImageNet-pretrained official checkpoint')
        assert args.method == 'simclr' and args.model == 'resnet50'
        if torch.cuda.device_count() > 1:
            model.backbone.module.load_ssl_official_weights()
        else:
            model.backbone.load_ssl_official_weights()
    else: 
        raise NotImplemented

    SAMPLING = {'random': random_sampling, 'simcore': simcore_sampling}
    selected_indices = SAMPLING[args.sampling_method](model, args)

    if args.stop:
        raw_epochs = copy.deepcopy(args.epochs)
        # re-calculate by the ratio to ImageNet length
        args.epochs = min(args.epochs, int(100 / (len(selected_indices) / 1281167)))
        if args.from_ssl_official: 
            args.epochs = int(args.epochs * 0.2) # fine-tuning the official pretrained model
        print('new epoch: {:d}'.format(args.epochs))
        
        if args.method == 'swav':
            args.freeze_prototypes = max(int((args.epochs * 10) / raw_epochs), 1)
            print('freeze prototypes: {}'.format(args.freeze_prototypes))
        if args.method == 'dino':
            if args.warm: args.warm_epochs = int((args.epochs * 100) / raw_epochs)
            args.freeze_last_layer = max(int((args.epochs * 10) / raw_epochs), 1)
            args.temp_warmup_epochs = int((args.epochs * 2000) / raw_epochs)
            model.temp_warmup_epochs = args.temp_warmup_epochs
            print('warm epochs: {}, freeze last layer: {}, temp warmup epochs: {}'.format(args.warm_epochs, args.freeze_last_layer, args.temp_warmup_epochs))
        if args.method == 'mae':
            if args.warm: args.warm_epochs = int((args.epochs * 100) / raw_epochs)
            print('warm epochs: {}'.format(args.warm_epochs))
    
    if args.sampling_times > 1: # multiple times of sampling coreset
        args.sampling_epochs = [int(args.epochs / args.sampling_times * (i+1)) for i in range(args.sampling_times-1)]
    else:
        args.sampling_epochs = []

    if not args.from_ssl_official:
        model.load_state_dict(init_ckpt) # initialize the model after openset sampling

    return selected_indices, model, args