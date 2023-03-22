TAG=""
DATA=""

# SimCore pretrained checkpoint
python train.py --tag $TAG \
    --dataset $DATA \
    --data_folder /path/to/data \
    --data_folder2 /path/to/data/WebFG-496 \
    --model resnet50 \
    --method simclr \
    --semisup_method self_training \
    --optimizer sgd \
    --learning_rate 0.03 \
    --weight_decay 1e-4 \
    --cosine \
    --total_steps 16384 \
    --eval_step 512 \
    --batch_size 256 \
    --label_ratio 0.5 \
    --mu 1 \
    --teacher_epochs 100 \
    --teacher_batch_size 256 \
    --T 1 \
    --lambda_u 0.5 \
    --save_freq 10  \
    --pretrained \
    --pretrained_ckpt /path/to/ckpt/last.pth

# train from scratch
python train.py --tag $TAG \
    --dataset $DATA \
    --data_folder /path/to/data \
    --data_folder2 /path/to/data/WebFG-496 \
    --model resnet50 \
    --method simclr \
    --semisup_method self_training \
    --optimizer sgd \
    --learning_rate 0.03 \
    --weight_decay 1e-4 \
    --cosine \
    --total_steps 65536 \
    --eval_step 512 \
    --batch_size 64 \
    --label_ratio 0.5 \
    --mu 1 \
    --teacher_epochs 500 \
    --teacher_batch_size 64 \
    --T 1 \
    --lambda_u 0.5 \
    --save_freq 30