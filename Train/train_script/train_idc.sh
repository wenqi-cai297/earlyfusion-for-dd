set -eux

ORIGINAL_PATH="path/to/ImageIDC"
DATASET="imagenet"
SUBSET="idc"

cd path/to/Folder/Train
accelerate launch train_cross_attn.py   \
  --seed 0   \
  --epoch 4   \
  --batch_size 16   \
  --dataset=$DATASET   --subset=$SUBSET    \
  --original_dataset_path=$ORIGINAL_PATH