TAG=""
DATA=""

python train.py --tag $TAG \
    --root /data/${DATA}/ \
    --dataset $DATA \
    --backbone resnet50 \
    --pretrained_backbone /path/to/ckpt/last.pth \
    --epochs 100 \
    --precision \
    --optimizer adam \
    --learning_rate 0.0001 \
    --cosine \
    --iou_thresh 0.2 \
    --class_label \
    --predict_path ./mAP/input/${TAG}/detection-results \
    --gt_bbox_path ./mAP/input/${TAG}/ground-truth

    # only for getting prediction bbox
    # --pretrained_ckpt /path/to/ckpt/last.pth 

IoU_start="0.5"
IoU_end="0.95"
Step="0.05"

for i in $(seq $IoU_start $Step $IoU_end)
do
    echo "IoU threshold is $i"
    python ./mAP/main.py --tag $TAG --dataset $DATA --iou_thresh $i
done
