#!/usr/bin/env bash
# Builds the LUMI convert container for Megatron-Bridge HF conversion:
#
#   the LUMI training container (config/container/lumi.yaml, built from
#   MegatronTrainingLumi.def.in) + nvidia-modelopt and pulp
#
# Megatron-Bridge's AutoBridge imports nvidia-modelopt at module level
# (auto_bridge.py), and modelopt imports pulp; the training container has neither.
# They are installed with --no-deps, so torch, transformer-engine and the rest of the
# training stack stay exactly as in the training container. Megatron-LM and
# Megatron-Bridge are not in the image: the convert jobs put submodules/Megatron-LM
# and submodules/Megatron-Bridge/src on PYTHONPATH.
#
# LUMI has neither fakeroot (/etc/subuid) nor proot, so a .def file can't be built
# there. Instead the image is unpacked into a writable sandbox on the login node's
# local /tmp (the unpacked image has too many files for the scratch quota),
# pip-installed into, and packed again. Needs no root; takes ~20 min.
#
# Usage (login node, from the repo root):
#   bash container/megatron/build_lumi_bridge_container.sh OUTPUT.sif [BASE.sif]
set -euo pipefail

OUT=${1:?usage: $0 OUTPUT.sif [BASE.sif]}
BASE=${2:-/scratch/project_465002530/containers/AutoExp_2026-07-02.sif}
PACKAGES=(nvidia-modelopt==0.47.0 pulp==3.3.2)

[ -f "$BASE" ] || { echo "error: base image $BASE not found" >&2; exit 1; }
[ ! -e "$OUT" ] || { echo "error: $OUT exists already" >&2; exit 1; }

SB=/tmp/$USER-bridge-sandbox-$$
trap 'rm -rf "$SB"' EXIT

echo "== unpacking $BASE into $SB"
singularity build --sandbox --fix-perms "$SB" "$BASE"

echo "== installing ${PACKAGES[*]}"
umask 022
singularity exec --writable --no-home --no-umask "$SB" \
    /opt/venv/bin/python3 -m pip install --no-cache-dir --no-deps "${PACKAGES[@]}"
# readable for every user of the image
chmod -R a+rX "$SB"/opt/venv/lib/python3.12/site-packages/{modelopt,pulp,nvidia_modelopt-*,pulp-*}

echo "== checking the imports"
singularity exec --no-home "$SB" \
    /opt/venv/bin/python3 -c "import modelopt.torch.quantization.utils, pulp; print('ok')"

echo "== packing $OUT"
singularity build "$OUT" "$SB"
echo "done: $OUT (base: $BASE)"
