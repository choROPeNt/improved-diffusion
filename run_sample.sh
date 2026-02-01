#!/usr/bin/env bash

RUN_ID="$(date +%Y-%m-%d_%H-%M)_$RANDOM"
OUT_DIR="/data/horse/ws/dchristi-diffusion/results/CT_Hys/${RUN_ID}"
mkdir -p "$OUT_DIR"

# --- Use arrays to avoid quoting/whitespace bugs ---
DATA_FLAGS=(
  --file_list ./data/file_list_sample.txt
  --dataset_type hdf5
  --dir "$OUT_DIR"
  --dataset_key patch_lr
)



MODEL_FLAGS=(
  --in_channel 1
  --dims 3
  --spatial_size 128
  --num_channels 64
  --num_res_blocks 2
  --learn_sigma true
  --class_cond false
  --model_path /data/horse/ws/dchristi-diffusion/checkpoints/CT_Hys/20260131_120116_11752/model030000.pt
)

DIFFUSION_FLAGS=(
  --diffusion_steps 4000
  --noise_schedule linear
  --rescale_learned_sigmas false
  --rescale_timesteps false
)

SAMPLE_FLAGS=(
  --batch_size 1
  --num_samples 1
)

echo "OUT_DIR=$OUT_DIR"
echo "DATA_FLAGS: ${DATA_FLAGS[*]}"
echo "MODEL_FLAGS: ${MODEL_FLAGS[*]}"
echo "DIFFUSION_FLAGS: ${DIFFUSION_FLAGS[*]}"
echo "SAMPLE_FLAGS: ${SAMPLE_FLAGS[*]}"

python scripts/super_res_sample.py \
  "${DATA_FLAGS[@]}" \
  "${MODEL_FLAGS[@]}" \
  "${DIFFUSION_FLAGS[@]}" \
  "${SAMPLE_FLAGS[@]}"