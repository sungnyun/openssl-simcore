# MixMatch template -> for other methods, please change semisup_method and modify related arguments to default values
TAG=""
DATA=""

python train.py --tag $TAG \
    --cosine \
    --total_step 16384 \
    --eval_step 512 \
    --dataset $DATA \
    --data_folder /path/to/data \
    --method mixmatch \
    --pretrained \
    --pretrained_ckpt /path/to/ckpt \
    --learning_rate 3e-2 \
    --weight_decay 1e-4 \
    --T 0.5 \
    --mixup_beta 0.75 \
    --lambda_u 75