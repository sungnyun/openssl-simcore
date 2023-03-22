# self-supervised learning with coreset sampling
TAG=""
DATA=""

python train_selfsup.py --tag $TAG \
    --merge_dataset \
    --model resnet50 \
    --batch_size 512 \
    --precision \
    --dataset1 $DATA \
    --dataset2 imagenet \
    --data_folder1 /path/to/data \
    --data_folder2 /path/to/data/ILSVRC2015/ILSVRC2015/Data/CLS-LOC/ \
    --method simclr \
    --epochs 5000 \
    --cosine \
    --optimizer sgd \
    --learning_rate 1e-1 \
    --weight_decay 1e-4  \
    --sampling_method simcore \
    --retrieval_ckpt /path/to/retrieval_ckpt/last.pth \
    --cluster_num 100 \
    --stop \
    --stop_thresh 0.95


# vanilla self-supervised learning without sampling
TAG=""
DATA=""

python train_selfsup.py --tag $TAG \
    --no_sampling \
    --model resnet50 \
    --batch_size 512 \
    --precision \
    --dataset $DATA \
    --data_folder /path/to/data \
    --method simclr \
    --epochs 5000 \
    --cosine \
    --learning_rate 1e-1 \
    --weight_decay 1e-4
