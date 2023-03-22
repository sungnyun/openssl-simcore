TAG=""
DATA=""

# SimCore pretrained checkpoint
python train.py --tag $TAG \
    --dataset $DATA \
    --data_folder /path/to/data \
    --data_folder2 /path/to/data/WebFG-496 \
    --model resnet50 \
    --method simclr \
    --optimizer sgd \
    --learning_rate 0.03 \
    --weight_decay 1e-4 \
    --cosine \
    --epochs 400 \
    --batch_size 128 \
    --noise_method dividemix \
    --warmup 30 \
    --alpha 0.75 \
    --lambda_u 75 \
    --p_threshold 0.2 \
    --T 0.5 \
    --pretrained \
    --pretrained_ckpt /path/to/ckpt/last.pth

# train from scratch
python train.py --tag $TAG \
    --dataset $DATA \
    --data_folder /path/to/data \
    --data_folder2 /path/to/data/WebFG-496 \
    --model resnet50 \
    --method simclr \
    --optimizer sgd \
    --learning_rate 0.03 \
    --weight_decay 1e-4 \
    --cosine \
    --epochs 400 \
    --batch_size 128 \
    --noise_method dividemix \
    --warmup 30 \
    --alpha 0.75 \
    --lambda_u 75 \
    --p_threshold 0.2 \
    --T 0.5
