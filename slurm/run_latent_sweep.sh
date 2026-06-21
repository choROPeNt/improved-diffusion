#!/usr/bin/env bash
# Latent channels sweep for VAE training.
# Creates a temporary config per latent_channels value and launches vae_train.py.
# out_dir is auto-derived from the YAML interpolation:
#   experiments/vae/lc${model.latent_channels}_b${objective.beta}

set -euo pipefail

CONFIG="configs/vae_train.yaml"
LATENT_CHANNELS=(2 4 8 16 32)

for lc in "${LATENT_CHANNELS[@]}"; do
    tmp=$(mktemp /tmp/vae_cfg_XXXXXX.yaml)
    sed "s/^\( *latent_channels:\s*\)[^ ]*/\1${lc}/" "$CONFIG" > "$tmp"

    echo "=== latent_channels=${lc}  config=${tmp} ==="
    python scripts/vae_train.py --config "$tmp"

    rm -f "$tmp"
done
