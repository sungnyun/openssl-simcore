# HardSampling template -> for other methods, please modify method and related arguments to default values
TAG=""
DATA=""

python train.py --tag $TAG \
    --dataset1 $DATA \
    --dataset2 imagenet \
    --data_folder1 /path/to/data \
    --data_folder2 /path/to/data/ILSVRC2015/ILSVRC2015/Data/CLS-LOC/ \
    --model resnet18 \
    --cosine \
    --precision \
    --method sampling \
    --epochs 5000 \
    --beta 1.0 \
    --tau_plus 0.01