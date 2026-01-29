#!/usr/bin/env bash

DATA_FLAGS="\
--data_dir ./data \
--dir /data/horse/ws/dchristi-diffusion/checkpoints/test \
"

MODEL_FLAGS="       \
--in_channel 3      \
--dims 2            \
--num_channels 192  \
--num_res_blocks 2  \
--learn_sigma true  \
--class_cond true   \
"

DIFFUSION_FLAGS="               \
--diffusion_steps 4000          \
--noise_schedule linear         \
--rescale_learned_sigmas false  \
--rescale_timesteps false
"

TRAIN_FLAGS="       \
--lr 3e-4           \
--batch_size 4
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