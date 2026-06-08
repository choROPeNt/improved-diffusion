#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=capella
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -J "VAE_sweep"
#SBATCH --output=out/VAE_sweep-%A_%a.out
#SBATCH --mail-user=christian.duereth@tu-dresden.de
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
#SBATCH -A p_haftfaeden
#SBATCH --array=0-5
#
# Latent-dimension sweep on a single H100. Each array task trains one
# latent_channels value. For a beta sweep instead, swap LATENT_DIMS/BETAS below
# and index BETAS with $SLURM_ARRAY_TASK_ID.
#
#   sbatch slurm/train_vae.sbatch
#
# Single run (no array): comment out the #SBATCH --array line and set IDX=0.

set -euo pipefail

ml release/24.10 GCC/13.3.0 Python/3.12.3 CUDA/12.8.0 OpenMPI/5.0.3
source /data/horse/ws/dchristi-diffusion/.venv/bin/activate

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

# ── sweep grids ─────────────────────────────────────────────────────────────
LATENT_DIMS=(2 4 8 16 32 48)          # --array=0-5
BETA=5e-2                              # fixed beta for the latent-dim sweep
IDX="${SLURM_ARRAY_TASK_ID:-0}"
LC="${LATENT_DIMS[$IDX]}"

DATA_PATH=/data/horse/ws/dchristi-diffusion/data/patches_vae.h5
RUN_ROOT=/data/horse/ws/dchristi-diffusion/experiments/vae_latentdim
OUT_DIR="${RUN_ROOT}/lc${LC}_b${BETA}"
mkdir -p "$OUT_DIR" out

echo "IDX=$IDX  latent_channels=$LC  beta=$BETA  -> $OUT_DIR"

# Hyperparameters come from configs/vae_train.yaml; here we only override the
# per-task pieces (latent_channels, beta, paths). CLI flags win over the config.
srun python scripts/vae_train.py \
  --config configs/vae_train.yaml \
  --data_path "$DATA_PATH" \
  --out_dir   "$OUT_DIR" \
  --latent_channels "$LC" \
  --beta "$BETA"
