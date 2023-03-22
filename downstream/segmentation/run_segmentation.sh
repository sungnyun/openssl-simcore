# on pets or cub dataset
python train.py \
    --backbone resnet50 \
    --lr 1e-1 \
    --weight-decay 1e-4 \
    --nesterov \
    --epochs 30 \
    --workers 16 \
    --gpu-ids 0,1,2,3 \
    --checkname deeplab-resnet50 \
    --eval-interval 30 \
    --dataset pets \
    --root /path/to/data \
    --pretrained-backbone /path/to/ckpt/last.pth
