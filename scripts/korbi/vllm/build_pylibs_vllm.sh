#!/bin/bash
# vLLM for MareNostrum 5 (offline): the release wheel and its dependencies from a wheel directory
# on GPFS (downloaded elsewhere: pip download vllm==<ver> --platform manylinux_2_*_x86_64
# --python-version 3.12 --only-binary=:all:), installed into NODE-LOCAL $DEST (default
# /tmp/pylibs_vllm_<submodule sha>; ~2 min), then the Python files of submodules/vllm (the
# OpenEuroLLM fork, our model ports) over the installed package -- the release's compiled kernels
# stay (the fork sits on the release tag; Python-only changes). Not on GPFS: the tree has tens of
# thousands of small files and /gpfs/projects wrote ~45 of them a minute (2026-11-01). Jobs call
# it first and put $DEST FIRST on PYTHONPATH (its torch 2.13 replaces the image's).
#
#   in a job, from a synced snapshot:
#   DEST=/tmp/pylibs_vllm scripts/korbi/vllm/build_pylibs_vllm.sh [<wheel dir>]
set -euo pipefail
R=${R:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
W=${1:-$R/wheels_vllm032}
cd "$(dirname "$0")/../../.."
SRC=$PWD/submodules/vllm
SHA=$(grep -m1 "^submodules/vllm " SYNC_STAMP 2>/dev/null | awk '{print $3}')
SHA=${SHA:-$(git -C "$SRC" rev-parse --short HEAD)}
DEST=${DEST:-/tmp/pylibs_vllm_$SHA}
VER=$(ls "$W"/vllm-*.whl | head -1 | sed -E 's|.*/vllm-([0-9.]+)-.*|\1|')
if [ ! -f "$DEST/.installed_$VER" ]; then
    rm -rf "$DEST"; mkdir -p "$DEST.pip"
    TMPDIR=$DEST.pip python3 -m pip install --no-index --find-links "$W" --target "$DEST" "vllm==$VER" \
        > "$DEST.log" 2>&1 || { tail -30 "$DEST.log"; exit 1; }
    rm -rf "$DEST.pip"
    touch "$DEST/.installed_$VER"
fi
# our Python on top (no binaries in the source tree; nothing deleted from the install)
tar -C "$SRC/vllm" --exclude='*.so' --exclude=__pycache__ -cf - . | tar -C "$DEST/vllm" -xf -
# hides the image's optional packages that cannot load next to this torch (see the file)
cp scripts/korbi/vllm/sitecustomize.py "$DEST/sitecustomize.py"
echo "$SHA" > "$DEST/VLLM_COMMIT"
python3 - "$DEST" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import torch, vllm
from vllm.model_executor.models.registry import _TEXT_GENERATION_MODELS as m
print("vllm", vllm.__version__, vllm.__file__, "torch", torch.__version__, "cuda", torch.version.cuda,
      "oellm_ambitious registered:", "OellmAmbitiousForCausalLM" in m)
PY
echo "built: $DEST (vllm $VER + submodule $SHA)"
