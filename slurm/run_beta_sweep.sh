#!/usr/bin/env bash
# Beta sweep for VAE training.
# Creates a temporary config per beta value and launches vae_train.py.
# out_dir is auto-derived from the YAML interpolation:
#   experiments/vae/lc${model.latent_channels}_b${objective.beta}

set -euo pipefail

CONFIG="configs/vae_train.yaml"
BETAS=(0.001 0.01 0.05 0.1 0.5 1.0)



for beta in "${BETAS[@]}"; do
    tmp=$(mktemp /tmp/vae_cfg_XXXXXX.yaml)
    sed "s/^\( *beta:\s*\)[^ ]*/\1${beta}/" "$CONFIG" > "$tmp"

    echo "=== beta=${beta}  config=${tmp} ==="
    python scripts/vae_train.py --config "$tmp"

    rm -f "$tmp"
done
