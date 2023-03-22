# Random template -> for other methods, please change active_method
SYM="SimCore"  # X / OS
DATA="pets"  # cub / aircraft / cars
TAG="bs256_lr3e-2_ep100_ar0_${SYM}"
ALGO="random"

python train.py --tag $TAG \
    --dataset $DATA \
    --data_folder /path/to/data \
    --model resnet50 \
    --method simclr \
    --optimizer sgd \
    --learning_rate 3e-2 \
    --weight_decay 1e-4 \
    --epochs 100 \
    --label_ratio 0.1 \
    --batch_size 256 \
    --active_method $ALGO \
    --active_round 0 \
    --pretrained_ckpt /path/to/ckpt/last.pth

LABEL="0.2 0.3 0.4"
cnt=0
for l in $LABEL
do
    TAG="bs256_lr3e-2_ep100_ar${cnt}_${SYM}"
    CKPT="./save/${DATA}_resnet50_${ALGO}_active_${TAG}/best.pth"
    
    ((cnt++))
    TAG="bs256_lr3e-2_ep100_ar${cnt}_${SYM}"
    
    python train.py --tag $TAG \
        --dataset $DATA \
        --data_folder /path/to/data \
        --model resnet50 \
        --method simclr \
        --optimizer sgd \
        --learning_rate 3e-2 \
        --weight_decay 1e-4 \
        --epochs 100 \
        --label_ratio $l \
        --batch_size 256 \
        --active_method $ALGO \
        --active_round $cnt \
        --pretrained_ckpt ${CKPT}
done