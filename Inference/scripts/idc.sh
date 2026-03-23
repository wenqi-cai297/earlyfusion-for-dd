

nvidia-smi

export CUDA_VISIBLE_DEVICES=1
IMAGENET_FOLDER=Path/To/ImageIDC

cd ..

IPC=10
T_SG=25 # STOP GUIDANCE
SPEC=idc

for ((i=0; i < 3; i++))
do

OUTPUT_DATASET=results/ours/dit-distillation-sampling-t-$T_SG/$SPEC-$i-IPC-$IPC
TRAIN_SAVE_DIR=results/ours/train-$SPEC-mode-sampling-t-$T_SG/$SPEC-$i-IPC-$IPC


python sample_mode_guidance_with_attn.py --model DiT-XL/2 --image-size 256 \
 --save-dir $OUTPUT_DATASET --spec $SPEC --num-samples $IPC --guidance \
 --stop_t $STOP_T $T_SG --imagenet_dir $IMAGENET_FOLDER --seed $i --num-datasets 1

python train.py -d imagenet --imagenet_dir $OUTPUT_DATASET/dataset_0 $IMAGENET_FOLDER \
    -n resnet_ap --nclass 10 --norm_type instance --ipc $IPC --tag test --slct_type random --spec $SPEC --repeat 3 \
    --save-dir $TRAIN_SAVE_DIR-resnet_ap
done


IPC=20
T_SG=25 # STOP GUIDANCE
SPEC=idc

for ((i=0; i < 3; i++))
do

OUTPUT_DATASET=results/ours/dit-distillation-sampling-t-$T_SG/$SPEC-$i-IPC-$IPC
TRAIN_SAVE_DIR=results/ours/train-$SPEC-mode-sampling-t-$T_SG/$SPEC-$i-IPC-$IPC



python sample_mode_guidance_with_attn.py --model DiT-XL/2 --image-size 256 \
 --save-dir $OUTPUT_DATASET --spec $SPEC --num-samples $IPC --guidance \
 --stop_t $STOP_T $T_SG --imagenet_dir $IMAGENET_FOLDER --seed $i --num-datasets 1

python train.py -d imagenet --imagenet_dir $OUTPUT_DATASET/dataset_0 $IMAGENET_FOLDER \
    -n resnet_ap --nclass 10 --norm_type instance --ipc $IPC --tag test --slct_type random --spec $SPEC --repeat 3 \
    --save-dir $TRAIN_SAVE_DIR-resnet_ap
done



IPC=50
T_SG=25 # STOP GUIDANCE
SPEC=idc

for ((i=0; i < 3; i++))
do

OUTPUT_DATASET=results/ours/dit-distillation-sampling-t-$T_SG/$SPEC-$i-IPC-$IPC
TRAIN_SAVE_DIR=results/ours/train-$SPEC-mode-sampling-t-$T_SG/$SPEC-$i-IPC-$IPC


python sample_mode_guidance_with_attn.py --model DiT-XL/2 --image-size 256 \
 --save-dir $OUTPUT_DATASET --spec $SPEC --num-samples $IPC --guidance \
 --stop_t $STOP_T $T_SG --imagenet_dir $IMAGENET_FOLDER --seed $i --num-datasets 1

python train.py -d imagenet --imagenet_dir $OUTPUT_DATASET/dataset_0 $IMAGENET_FOLDER \
    -n resnet_ap --nclass 10 --norm_type instance --ipc $IPC --tag test --slct_type random --spec $SPEC --repeat 3 \
    --save-dir $TRAIN_SAVE_DIR-resnet_ap
done
