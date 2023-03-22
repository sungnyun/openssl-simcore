TAG=""
DATA=""

# SimCore pretrained checkpoint
python train.py --tag $TAG \
    --dataset $DATA \
    --data_folder /path/to/data \
    --data_folder2 /path/to/data/WebFG-496 \
    --model resnet50 \
    --method simclr \
    --semisup_method openmatch \
    --optimizer sgd \
    --learning_rate 0.03 \
    --weight_decay 1e-4 \
    --cosine \
    --start_fix 5 \
    --total_step 16384 \
    --eval_step 512 \
    --batch_size 64 \
    --label_ratio 0.5 \
    --mu 2 \
    --T 1 \
    --lambda_oem 0.1 \
    --lambda_socr 0.5 \
    --threshold 0.0 \
    --save_freq 10 \
    --pretrained \
    --pretrained_ckpt /path/to/ckpt/last.pth

# train from scratch
python train.py --tag $TAG \
    --dataset $DATA \
    --data_folder /path/to/data \
    --data_folder2 /path/to/data/WebFG-496 \
    --model resnet50 \
    --method simclr \
    --semisup_method openmatch \
    --optimizer sgd \
    --learning_rate 0.03 \
    --weight_decay 1e-4 \
    --cosine \
    --start_fix 20 \
    --total_step 65536 \
    --eval_step 512 \
    --batch_size 64 \
    --label_ratio 0.5 \
    --mu 2 \
    --T 1 \
    --lambda_oem 0.1 \
    --lambda_socr 0.5 \
    --threshold 0.0 \
    --save_freq 30
