#!/bin/bash
NUM_GPUS=${1:-1}

sbatch \
    --time=00:01:00 \
    --nodes=1 \
    --ntasks=${NUM_GPUS} \
    --gres=gpu:${NUM_GPUS} \
    --cpus-per-task=6 \
    --partition=capella \
    --mem-per-cpu=12G \
    --job="Diffusion_super_res" \
    --output=out/diff_super-res-%j.out \
    --mail-user=christian.duereth@tu-dresden.de \
    --account=p_biiax
    train_super-res.sbatch






