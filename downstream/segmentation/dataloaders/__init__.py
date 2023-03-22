# from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd
from dataloaders.datasets import pets_segmentation, cub_segmentation
from torch.utils.data import DataLoader


def make_data_loader(args, **kwargs):
    if args.dataset == 'pets':
        # root = '~~~/pets/'
        train_set = pets_segmentation.PetsSegmentationDataset(args.root, test=False, class_label=args.class_label)
        val_set = pets_segmentation.PetsSegmentationDataset(args.root, test=True, class_label=args.class_label)

        # class: foreground, background, unknown (boundary is not considered)
        num_class = 2 if not args.class_label else 38
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cub':
        # root = '~~~/cub/'
        train_set = cub_segmentation.CUBSegmentationDataset(args.root, test=False, class_label=args.class_label)
        val_set = cub_segmentation.CUBSegmentationDataset(args.root, test=True, class_label=args.class_label)

        # class: background, foreground (5/5 experts coincide), unknown (below 4/5 experts coincide)
        num_class = 2 if not args.class_label else 201
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    # if args.dataset == 'pascal':
    #     train_set = pascal.VOCSegmentation(args, split='train')
    #     val_set = pascal.VOCSegmentation(args, split='val')
    #     if args.use_sbd:
    #         sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
    #         train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

    #     num_class = train_set.NUM_CLASSES
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    #     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    #     test_loader = None

    #     return train_loader, val_loader, test_loader, num_class

    # elif args.dataset == 'cityscapes':
    #     train_set = cityscapes.CityscapesSegmentation(args, split='train')
    #     val_set = cityscapes.CityscapesSegmentation(args, split='val')
    #     test_set = cityscapes.CityscapesSegmentation(args, split='val')
    #     num_class = train_set.NUM_CLASSES
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    #     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    #     test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

    #     return train_loader, val_loader, test_loader, num_class

    # elif args.dataset == 'coco':
    #     train_set = coco.COCOSegmentation(args, split='train')
    #     val_set = coco.COCOSegmentation(args, split='val')
    #     test_set = coco.COCOSegmentation(args, split='val')
    #     num_class = train_set.NUM_CLASSES
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    #     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    #     test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    #     return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError

