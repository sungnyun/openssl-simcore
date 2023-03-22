TAG=""
DATA=""

python train_sup.py --tag $TAG \
    --dataset $DATA \
    --model resnet50 \
    --data_folder /path/to/data \
    --pretrained \
    --pretrained_ckpt /path/to/ckpt/last.pth
    --method simclr \
    --epochs 100 \
    --learning_rate 10 \
    --weight_decay 0

# knn (Table 7a)
# --knn \
# --topk 20 200

# semisup (Table 7b)
# --label_ratio 0.1 \
# --e2e