#!/bin/bash
# LEGACY (smoke tests 01-04 only: old Megatron fork + nemo_26.04 container).
# For current runs use ../setup_jupiter_hybrid.sh and ../README.md.
#
# One-time JUPITER setup for the opt_scale runs. Run on a LOGIN node (needs internet).
# Idempotent: skips what already exists, never deletes anything.
#   bash config/experiments/laingsam1/opt_scale/smoke_test/setup_jupiter.sh
#
# Creates:
#   $MEG    Megatron fork oellm/v0.19-moe-lumi-angular-muown @ 1f88af5d (GDN + AngularMuown)
#   $PYEXT  emerging_optimizers 0.2.0 (the container ships 0.1.0, which the fork can't use:
#           its import is wrapped in try/except, so Muon/AngularMuown would silently vanish)
# fla (flash-linear-attention) is already in the container (0.4.2) -> nothing to install.
set -euo pipefail

BASE=/e/scratch/e-aif-2026sc04-211/users/slaing
SIF=/e/project1/e-ext-2025e02-108/container/nemo_26.04.sif
MEG=$BASE/NVIDIA-Megatron-LM-angular
PYEXT=$BASE/pyext
COMMIT=1f88af5d140b2ac196df42fb9509755563c0adef

if [ ! -d "$MEG/.git" ]; then
  git clone -b oellm/v0.19-moe-lumi-angular-muown \
    https://github.com/OpenEuroLLM/NVIDIA-Megatron-LM.git "$MEG"
fi
if [ "$(git -C "$MEG" rev-parse HEAD)" != "$COMMIT" ]; then
  git -C "$MEG" checkout --detach "$COMMIT"
fi
echo "Megatron fork: $(git -C "$MEG" rev-parse --short HEAD) at $MEG"

if [ ! -d "$PYEXT/emerging_optimizers-0.2.0.dist-info" ]; then
  mkdir -p "$PYEXT"
  apptainer exec "$SIF" /opt/venv/bin/python -m pip install --no-deps --target "$PYEXT" \
    "git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.2.0"
fi
echo "emerging_optimizers 0.2.0 at $PYEXT"

mkdir -p "$BASE/logs" "$BASE/cache/megatron_index" "$BASE/experiments/opt_scale/smoke_test"
echo "Setup done. Next: bash config/experiments/laingsam1/opt_scale/smoke_test/00_env_check.sh"
