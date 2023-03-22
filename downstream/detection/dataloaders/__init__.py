import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
from dataloaders.datasets import aircraft_detection, cars_detection


def make_data_loader(args,  **kwargs):
    def collate_fn(batch):
        return tuple(zip(*batch))

    train_transform = A.Compose([
        # A.Resize(height=224, width=224, always_apply=True),
        A.BBoxSafeRandomCrop(),
        A.HorizontalFlip(p=0.5),
        ToTensorV2()
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

    test_transform = A.Compose([
        ToTensorV2()
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

    if args.dataset == 'aircraft':
        # root = '~~~/aircraft/'
        train_set = aircraft_detection.AircraftDetectionDataset(args.root, 'trainval', transforms=train_transform, class_label=args.class_label)
        test_set = aircraft_detection.AircraftDetectionDataset(args.root, 'test', transforms=test_transform, class_label=args.class_label)

        # 1 class (aircraft) + background
        num_class = 2 if not args.class_label else 101
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8, **kwargs)
        # test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn, **kwargs)

        return train_loader, test_set, num_class

    elif args.dataset == 'cars':
        # root = '~~~/cars/'
        train_set = cars_detection.CarDetectionDataset(args.root, split='train', transforms=train_transform, class_label=args.class_label)
        test_set = cars_detection.CarDetectionDataset(args.root, split='test', transforms=test_transform, class_label=args.class_label)

        # class: cars (localization)
        num_class = 2 if not args.class_label else 197
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8, **kwargs)
        # test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn, **kwargs)

        return train_loader, test_set, num_class

    else:
        raise NotImplementedError

