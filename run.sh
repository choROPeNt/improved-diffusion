#!/usr/bin/env bash

RUN_ID="$(date +%Y%m%d_%H%M%S)_$RANDOM"
OUT_DIR="/data/horse/ws/dchristi-diffusion/checkpoints/CT_Hys/${RUN_ID}"

DATA_FLAGS="\
--file_list ./data/file_list.txt \
--dataset_type hdf5 \
--dir ${OUT_DIR} \
"

# --data_dir ./data/super_res_test/hr \


MODEL_FLAGS="       \
--in_channel 1      \
--dims 3            \
--spatial_size 128  \
--num_channels 64   \
--num_res_blocks 2  \
--learn_sigma true  \
--class_cond false  \
"

DIFFUSION_FLAGS="               \
--diffusion_steps 4000          \
--noise_schedule linear         \
--rescale_learned_sigmas false  \
--rescale_timesteps false
"

TRAIN_FLAGS="       \
--lr 3e-4           \
--batch_size 1
"

printf "DATA_FLAGS:\n%s\n" "$DATA_FLAGS"
printf "MODEL_FLAGS:\n%s\n" "$MODEL_FLAGS"
printf "DIFFUSION_FLAGS:\n%s\n" "$DIFFUSION_FLAGS"
printf "TRAIN_FLAGS:\n%s\n" "$TRAIN_FLAGS"

python scripts/super_res_train.py   \
    $DATA_FLAGS                     \
    $MODEL_FLAGS                    \
    $DIFFUSION_FLAGS                \
    $TRAIN_FLAGS