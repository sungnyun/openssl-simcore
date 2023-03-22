import os

import torch
import torch.nn as nn
from torchvision import transforms, datasets


# refer to https://github.com/Reza-Safdari/SimSiam-91.9-top1-acc-on-CIFAR10/blob/f7365684399b1e6895f81ff059eaa4e2b8c73608/simsiam/validation.py#L9
class KNNValidation(object):
    DATASET_CONFIG = {'cars': 196, 'flowers': 102, 'pets': 37, 'aircraft': 100, 'cub': 200, 'dogs': 120,
                      'mit67': 67, 'stanford40': 40, 'dtd': 47, 'imagenet100': 100, 'imagenet': 1000}
    def __init__(self, args):
        self.args = args
        
        normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        base_transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             normalize])

        if args.dataset in self.DATASET_CONFIG:
            if args.dataset == 'imagenet':
                traindir = os.path.join(args.data_folder, 'train') # under ~~/Data/CLS-LOC
                valdir = os.path.join(args.data_folder, 'val')
            else:
                traindir = os.path.join(args.data_folder, args.dataset, 'train')
                valdir = os.path.join(args.data_folder, args.dataset, 'test')
            train_dataset = datasets.ImageFolder(root=traindir, transform=base_transform)
            val_dataset = datasets.ImageFolder(root=valdir, transform=base_transform)
        else:
            raise NotImplementedError

        # shuffle must be False
        self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)
        
    def topk_retrieval(self, model):
        model.eval()
        
        n_data = len(self.train_loader.dataset)
        train_features = torch.zeros([model.module.final_feat_dim, n_data])

        top1_list = []
        for topk in self.args.topk:
            with torch.no_grad():
                for idx, (images, _) in enumerate(self.train_loader):
                    images = images.cuda(non_blocking=True)
                    bsz = images.shape[0]

                    # forward
                    features = model(images)
                    features = nn.functional.normalize(features)
                    train_features[:, bsz*idx : bsz*idx+bsz] = features.data.t()

                train_labels = torch.LongTensor(self.train_loader.dataset.targets)

            total = 0
            correct = 0
            with torch.no_grad():
                for idx, (images, labels) in enumerate(self.val_loader):
                    images = images.cuda(non_blocking=True)
                    # labels = labels.cuda(non_blocking=True)
                    bsz = images.shape[0]

                    features = model(images)
                    features = nn.functional.normalize(features)
                    dist = torch.mm(features.cpu(), train_features)
                    
                    # top-k
                    yd, yi = dist.topk(topk, dim=1, largest=True, sorted=True)
                    candidates = train_labels.view(1, -1).expand(bsz, -1)
                    retrieval = torch.gather(candidates, 1, yi)
                    # retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)    
                                
                    weight = torch.exp(yd / 0.07)  # use temperature 0.07

                    preds = []
                    for i, ret in enumerate(retrieval):
                        unique = {cls.item(): 0 for cls in ret.unique()}
                        for r, w in zip(ret, weight[i]):
                            unique[r.item()] += w.item()
                        pred, v = sorted(unique.items(), key=lambda item: item[1], reverse=True)[0]
                        preds.append(pred)
                    preds = torch.tensor(preds)

                    total += labels.size(0)
                    correct += preds.eq(labels.data).sum().item()
                top1 = round(correct / total * 100, 2)

            print(' * {topk}-NN Acc@1 {top1:.2f}'.format(topk=topk, top1=top1))
            top1_list.append(top1)
            
        return top1_list
    