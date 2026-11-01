#!/bin/bash
# vLLM for MareNostrum 5 (offline): the release wheel and its dependencies from a wheel directory
# (downloaded elsewhere: pip download vllm==<ver> --platform manylinux_2_*_x86_64 --python-version 3.12
# --only-binary=:all:), installed into $R/pylibs_vllm_<submodule sha>, then the Python files of
# submodules/vllm (the OpenEuroLLM fork, our model ports) over the installed package -- the
# release's compiled kernels stay (the fork sits on the release tag; Python-only changes). Not
# into the synced snapshot: a resync would delete the installed binaries. Configs put the
# directory FIRST on PYTHONPATH (its torch 2.13 replaces the image's).
#
#   run from a synced snapshot (in a job: the login node kills processes after 300 s of CPU):
#   scripts/korbi/vllm/build_pylibs_vllm.sh [<wheel dir>]          # -> $R/pylibs_vllm_<sha>
set -euo pipefail
R=${R:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
W=${1:-$R/wheels_vllm032}
cd "$(dirname "$0")/../../.."
SRC=$PWD/submodules/vllm
SHA=$(grep -m1 "^submodules/vllm " SYNC_STAMP 2>/dev/null | awk '{print $3}')
SHA=${SHA:-$(git -C "$SRC" rev-parse --short HEAD)}
DEST=${DEST:-$R/pylibs_vllm_$SHA}
VER=$(ls "$W"/vllm-*.whl | head -1 | sed -E 's|.*/vllm-([0-9.]+)-.*|\1|')
if [ ! -f "$DEST/.installed_$VER" ]; then
    # pip --target copies its tree file by file: onto node-local disk first (GPFS takes minutes per
    # thousand small files under pip's copy), then one tar copy to GPFS
    LOCAL=$(mktemp -d "${LOCAL_TMP:-/tmp}/vllm_build_XXXX")
    TMPDIR=$LOCAL python3 -m pip install --no-index --find-links "$W" --target "$LOCAL/target" "vllm==$VER" \
        > "$R/.vllm_build_$SHA.log" 2>&1 || { tail -30 "$R/.vllm_build_$SHA.log"; exit 1; }
    rm -rf "$DEST.tmp"; mkdir -p "$DEST.tmp"
    tar -C "$LOCAL/target" -cf - . | tar -C "$DEST.tmp" -xf -     # no rsync in the image
    rm -rf "$LOCAL" "$DEST"; mv "$DEST.tmp" "$DEST"
    touch "$DEST/.installed_$VER"
fi
# our Python on top (no binaries in the source tree; nothing deleted from the install)
tar -C "$SRC/vllm" --exclude='*.so' --exclude=__pycache__ -cf - . | tar -C "$DEST/vllm" -xf -
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
