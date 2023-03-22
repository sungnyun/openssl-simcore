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
    --epochs 200 \
    --batch_size 256 \
    --noise_method co_teaching \
    --forget_rate 0.2 \
    --num_gradual 10 \
    --exponent 1 \
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
    --epochs 1000 \
    --batch_size 256 \
    --noise_method co_teaching \
    --forget_rate 0.2 \
    --num_gradual 10 \
    --exponent 1
