#!/bin/bash
# One-time JUPITER setup for the opt_scale runs on the TEAM Megatron (oellm/v0.19-hybrid).
# Run on a JUPITER LOGIN node (needs internet), from the oellm-autoexp repo root:
#   export OPT_SCALE_BASE=/e/scratch/<project>/users/<you>
#   bash config/experiments/laingsam1/opt_scale/setup_jupiter_hybrid.sh
#
# Idempotent: skips what already exists, never deletes or overwrites anything.
# Everything it creates lives under $OPT_SCALE_BASE:
#   NVIDIA-Megatron-LM-hybrid/  team Megatron, detached at the pinned commit (read-only use)
#   pyext/                      emerging_optimizers 0.2.0 (the container ships 0.1.0, which
#                               Megatron silently ignores -> Muon/AngularMuown would vanish)
#   venvs/autoexp/              Python 3.12 venv with this repo installed editable (for
#                               scripts/run_autoexp.py on the login node)
#   logs/ cache/megatron_index/ experiments/   (Slurm does not create the logs dir itself)
# fla (flash-linear-attention) is already in the container -> nothing to install.
set -euo pipefail

: "${OPT_SCALE_BASE:?set OPT_SCALE_BASE=/e/scratch/<project>/users/<you> first}"
BASE=$OPT_SCALE_BASE
REPO=$(git rev-parse --show-toplevel)
SIF=/e/project1/e-sta-openeurollm/container/MegatronTraining-JUPITER-te218-fa3_aarch64_202608280932.sif
MEG=$BASE/NVIDIA-Megatron-LM-hybrid
PIN=013763961102c13bc8d34ee501e616429aa3e5c1
PYEXT=$BASE/pyext
VENV=$BASE/venvs/autoexp

[ -r "$SIF" ] || { echo "Cannot read $SIF -> need group e-sta-openeurollm (JuDoor) + a fresh login"; exit 1; }

# 1. Team Megatron at the pinned commit (detached HEAD: you never commit to or push this checkout)
if [ ! -d "$MEG/.git" ]; then
  git clone -b oellm/v0.19-hybrid https://github.com/OpenEuroLLM/NVIDIA-Megatron-LM.git "$MEG"
fi
if [ "$(git -C "$MEG" rev-parse HEAD)" != "$PIN" ]; then
  git -C "$MEG" checkout --detach "$PIN"
fi
echo "[1/4] Megatron: $(git -C "$MEG" rev-parse --short HEAD) at $MEG"

# 2. emerging_optimizers 0.2.0, installed with the container's own pip into $PYEXT
if [ ! -d "$PYEXT/emerging_optimizers-0.2.0.dist-info" ]; then
  mkdir -p "$PYEXT"
  apptainer exec "$SIF" /opt/venv/bin/python -m pip install --no-deps --target "$PYEXT" \
    "git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.2.0"
fi
echo "[2/4] emerging_optimizers 0.2.0 at $PYEXT"

# 3. autoexp venv (uv; install uv first if missing: curl -LsSf https://astral.sh/uv/install.sh | sh)
if [ ! -x "$VENV/bin/python" ]; then
  command -v uv >/dev/null || { echo "uv not found (see comment above)"; exit 1; }
  export UV_CACHE_DIR=${UV_CACHE_DIR:-$BASE/.cache/uv}   # keep the cache off \$HOME (small quota)
  uv venv --python /usr/bin/python3.12 "$VENV"
  VIRTUAL_ENV=$VENV uv pip install -e "$REPO"
fi
echo "[3/4] autoexp venv at $VENV"

# 4. Directories the jobs write to
mkdir -p "$BASE/logs" "$BASE/cache/megatron_index" "$BASE/experiments/opt_scale" "$BASE/hf_home"
echo "[4/4] logs / cache / experiments / hf_home under $BASE"

echo
echo "Setup done. Next (see opt_scale/README.md):"
echo "  bash config/experiments/laingsam1/opt_scale/smoke_test/05_hybrid_env_check.sh"
