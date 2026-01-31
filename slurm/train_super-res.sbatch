#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=capella
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH -J "Diff_super-res"
#SBATCH --output=out/Diff_super-res-%j.out
#SBATCH --mail-user=christian.duereth@tu-dresden.de
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
#SBATCH -A p_biiax

set -euo pipefail

ml release/24.10 GCC/13.3.0 Python/3.12.3 CUDA/12.8.0 OpenMPI/5.0.3

source /data/horse/ws/dchristi-diffusion/.venv/bin/activate
export OMPI_MCA_smsc=^knem

echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

RUN_ID="$(date +%Y%m%d_%H%M%S)_$RANDOM"
OUT_DIR="/data/horse/ws/dchristi-diffusion/checkpoints/CT_Hys/${RUN_ID}"
mkdir -p "$OUT_DIR"

# --- Use arrays to avoid quoting/whitespace bugs ---
DATA_FLAGS=(
  --file_list ./data/file_list.txt
  --dataset_type hdf5
  --dir "$OUT_DIR"
)

MODEL_FLAGS=(
  --in_channel 1
  --dims 3
  --spatial_size 128
  --num_channels 64
  --num_res_blocks 2
  --learn_sigma true
  --class_cond false
  --use_fp16 true
)

DIFFUSION_FLAGS=(
  --diffusion_steps 4000
  --noise_schedule linear
  --rescale_learned_sigmas false
  --rescale_timesteps false
)

TRAIN_FLAGS=(
  --lr 5e-5
  --batch_size 1
)

echo "OUT_DIR=$OUT_DIR"
echo "DATA_FLAGS: ${DATA_FLAGS[*]}"
echo "MODEL_FLAGS: ${MODEL_FLAGS[*]}"
echo "DIFFUSION_FLAGS: ${DIFFUSION_FLAGS[*]}"
echo "TRAIN_FLAGS: ${TRAIN_FLAGS[*]}"

# --- Debug: show rank -> GPU mapping (each rank sees 1 GPU after we set CVD) ---
mpiexec -n "${SLURM_NTASKS}" bash -lc '
  export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
  echo "rank=$OMPI_COMM_WORLD_RANK local_rank=$OMPI_COMM_WORLD_LOCAL_RANK host=$(hostname) CVD=$CUDA_VISIBLE_DEVICES"
  python - << "PY"
import os, torch
print(" torch cuda visible:", torch.cuda.device_count())
print(" current device:", torch.cuda.current_device() if torch.cuda.is_available() else None)
print(" device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
'

# --- Launch training (robust argv passing) ---
mpiexec -n "${SLURM_NTASKS}" bash -lc '
  export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
  exec "$@"
' bash python scripts/super_res_train.py \
  "${DATA_FLAGS[@]}" \
  "${MODEL_FLAGS[@]}" \
  "${DIFFUSION_FLAGS[@]}" \
  "${TRAIN_FLAGS[@]}"