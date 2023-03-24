## How to setup datasets
After downloading each dataset from the official links, set the image files as the `torchvision.datasets.ImageFolder` format.    
In other words, all images should be constructed as this way:

```
data/aircraft
├── train
│  ├── Boeing_717
│  │  ├── 1801653.jpg
│  │  ├── 1385089.jpg
│  │  ├── 0181712.jpg
│  │
│  ├── CRJ-900
│  ├── A380
│
├── test
│  ├── Boeing_717
```

We provide the python files for the ease of ImageFolder generation. Please carefully modify source and target paths in each python files, and run `[DATASET_NAME]_image_folder_generator.py` to set the right paths for train and test folders.

We also provide the full details of each data we used.

| Dataset name (in paper) | name (code) | # of data | Link |                                                                                                                                                  
|-------------------------|----------------|-----------|------|
| Aircraft (FGVC-Aircraft) | aircraft | 6,667 | https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/ |
| Cars (Stanford Cars) | cars | 8,144 | https://ai.stanford.edu/~jkrause/cars/car_dataset.html |
| Pet (Oxford-IIIT Pet) | pets | 3,680 | https://www.robots.ox.ac.uk/~vgg/data/pets/ |
| Birds (Caltech-UCSD Birds) | cub | 5,990 | https://www.vision.caltech.edu/datasets/cub_200_2011/ |
| Dogs (Stanford Dogs) | dogs | 12,000 | http://vision.stanford.edu/aditya86/ImageNetDogs/ |
| Flowers (Oxford 102 Flower) | flowers | 2,040 | https://www.robots.ox.ac.uk/~vgg/data/flowers/ |
| Actions (Stanford 40 Actions) | stanford40 | 4,000 | http://vision.stanford.edu/Datasets/40actions.html |
| Indoor (MIT-67 Indoor Scene) | mit67 | 5,360 | https://web.mit.edu/torralba/www/indoor.html |
| Textures (Describable Textures) | dtd | 3,760 | https://www.robots.ox.ac.uk/~vgg/data/dtd/ |
| Faces (CelebAMask-HQ) | celeba | 4,263 | https://github.com/switchablenorms/CelebAMask-HQ |
| Food (Food 11) | food11 | 13,296 | https://www.kaggle.com/datasets/trolukovich/food11-image-dataset |
|                         |            |  |                | 
| ImageNet | imagenet | 1,281,167 | https://www.image-net.org |
| Microsoft COCO | coco | 118,287 | https://cocodataset.org/#home |
| iNaturalist2021-mini | inaturalist | 500,000 | https://github.com/visipedia/inat_comp/tree/master/2021 |
| Places365 | places | 8,026,628 | https://ai.stanford.edu/~jkrause/cars/car_dataset.html |
| ALL | everything | 9,926,082 | - |
| WebVision | webvision | 2,446,037 | https://data.vision.ee.ethz.ch/cvl/webvision/dataset2017.html |
| WebFG-496 | webfg | 53,339 | https://github.com/NUST-Machine-Intelligence-Laboratory/weblyFG-dataset |
