#!/bin/bash
# Source this inside the pinned MN5 training image. Dependencies come from offline wheels.
# The shared-runtime successor of setup_vllm_uv.sh (node-local /tmp install): it replaces that file once
# check_vllm_runtime.sh has passed on a node; until then only the check uses it.
#
# The vLLM runtime lives on GPFS ($OELLM_VLLM_ROOT, default $R/vllm_runtime), shared by every node
# and job, and is never modified once published:
#   base_<id>   venv + pylibs_vllm: stock vllm==0.32.0 and its dependencies from $R/wheels_vllm032
#               (id: the wheel set), built once; typer removed (the wheel pins an older one, and
#               lighteval's CLI metadata needs the image's)
#   src_<id>    the fork: vllm/ = submodules/vllm/vllm's Python tree, the base's other files
#               (compiled extensions, data) as symlinks, + sitecustomize.py (id: the fork's whole
#               tree and sitecustomize.py, so every code change gets its own directory)
# Each is built in a private directory and published by one rename, under a lock directory, so a
# job never sees a half-built tree and concurrent jobs never overwrite each other's code. Only
# FlashInfer's JIT cache stays node-local (/tmp): sharing it, concurrent first builds raced
# (sitecustomize.py).
# Exports INFER_PY, DEST (the base's pylibs_vllm), OELLM_VLLM_SRC (the src_ directory), PYTHONPATH
# and the offline HF / NLTK environment.
set -euo pipefail
R=${R:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
UV_BIN=${OELLM_UV_BIN:-$R/tools/uv}
[ -x "$UV_BIN" ] || UV_BIN=$(command -v uv)
WHEELS=${OELLM_VLLM_WHEELS:-$R/wheels_vllm032}
VLLM_ROOT=${OELLM_VLLM_ROOT:-$R/vllm_runtime}
[ -d "$WHEELS" ] || { echo "[setup_vllm] no wheel directory $WHEELS" >&2; return 1 2>/dev/null || exit 1; }
mkdir -p "$VLLM_ROOT"

# _oellm_publish <final dir> <build function>: build into a private directory, rename it into place
# (the lock directory keeps concurrent jobs from building the same tree twice; a lock older than
# 60 min is a dead builder's)
_oellm_publish() {
 local final=$1 build=$2 lock="$1.lock" tmp waited=0
 while [ ! -d "$final" ]; do
  if mkdir "$lock" 2>/dev/null; then
   echo "$(hostname) $$ $(date -Is)" > "$lock/owner"
   # build directories of killed jobs
   find "$(dirname "$final")" -maxdepth 1 -name "$(basename "$final").tmp.*" -mmin +720 -exec rm -rf {} + 2>/dev/null || true
   if [ ! -d "$final" ]; then
    tmp="$final.tmp.$(hostname).$$"
    command rm -rf "$tmp"
    mkdir -p "$tmp"
    echo "[setup_vllm] building $final" >&2
    if ! "$build" "$tmp"; then
     command rm -rf "$tmp" "$lock"
     return 1
    fi
    mv -T "$tmp" "$final"
   fi
   command rm -rf "$lock"
  elif [ -n "$(find "$lock" -maxdepth 0 -mmin +60 2>/dev/null)" ]; then
   echo "[setup_vllm] removing stale lock $lock ($(cat "$lock/owner" 2>/dev/null))" >&2
   command rm -rf "$lock"
  else
   [ $((waited % 60)) -eq 0 ] && echo "[setup_vllm] waiting for $final ($(cat "$lock/owner" 2>/dev/null))" >&2
   sleep 5
   waited=$((waited + 5))
  fi
 done
}

_oellm_build_base() {
 local d=$1
 export UV_CACHE_DIR=$d/uv-cache
 "$UV_BIN" venv --system-site-packages --python /usr/bin/python3 "$d/.venv" > "$d/install.log" 2>&1
 mkdir -p "$d/pylibs_vllm"
 "$UV_BIN" pip install --no-index --find-links "$WHEELS" --python "$d/.venv/bin/python" \
  --target "$d/pylibs_vllm" 'vllm==0.32.0' >> "$d/install.log" 2>&1 || { tail -40 "$d/install.log" >&2; return 1; }
 command rm -rf "$d/uv-cache" "$d"/pylibs_vllm/typer "$d"/pylibs_vllm/typer-*.dist-info "$d"/pylibs_vllm/typer_slim-*.dist-info
 # the venv's python inside the published tree, not the private build path
 sed -i "s#$d#${d%.tmp.*}#g" "$d/.venv/pyvenv.cfg" "$d"/.venv/bin/activate* 2>/dev/null || true
}

_oellm_build_src() {
 local d=$1
 python3 - "$BASE/pylibs_vllm/vllm" submodules/vllm/vllm "$d/vllm" <<'EOF'
import os
import shutil
import sys

base, fork, out = sys.argv[1:]
for root, dirs, files in os.walk(fork):  # the fork's Python tree (its .so files are local builds)
    dirs[:] = [x for x in dirs if x != "__pycache__"]
    rel = os.path.relpath(root, fork)
    os.makedirs(os.path.join(out, rel), exist_ok=True)
    for f in files:
        if not f.endswith(".so"):
            shutil.copy2(os.path.join(root, f), os.path.join(out, rel, f))
for root, dirs, files in os.walk(base):  # everything else from the stock install, linked
    dirs[:] = [x for x in dirs if x != "__pycache__"]
    rel = os.path.relpath(root, base)
    os.makedirs(os.path.join(out, rel), exist_ok=True)
    for f in files:
        if not os.path.lexists(os.path.join(out, rel, f)):
            os.symlink(os.path.join(root, f), os.path.join(out, rel, f))
EOF
 # hides the image's optional packages that cannot load next to this torch (see the file)
 cp scripts/korbi/vllm/sitecustomize.py "$d/sitecustomize.py"
 printf '%s\n' "$VLLM_SOURCE_ID" > "$d/RUNTIME_SOURCE_ID"
}

BASE_ID=$( { echo 'vllm==0.32.0 typer-pruned'; /usr/bin/python3 --version; ls -l "$WHEELS" | awk '{print $5, $NF}'; } \
 | sha256sum | cut -c1-16)
BASE=$VLLM_ROOT/base_$BASE_ID
_oellm_publish "$BASE" _oellm_build_base || return 1 2>/dev/null || exit 1
VLLM_SOURCE_ID=$( { tar -C submodules/vllm/vllm --exclude='*.so' --exclude=__pycache__ --sort=name --mtime=@0 \
 --owner=0 --group=0 --numeric-owner -cf - .; cat scripts/korbi/vllm/sitecustomize.py; echo "$BASE_ID"; } \
 | sha256sum | cut -c1-16)
export OELLM_VLLM_SRC=$VLLM_ROOT/src_$VLLM_SOURCE_ID
_oellm_publish "$OELLM_VLLM_SRC" _oellm_build_src || return 1 2>/dev/null || exit 1
export INFER_PY=$BASE/.venv/bin/python
export DEST=$BASE/pylibs_vllm
# FlashInfer's JIT builds: node-local, one per node and base install (sitecustomize.py)
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-/tmp/oellm_flashinfer_$BASE_ID}
mkdir -p "$FLASHINFER_WORKSPACE_BASE"

export PYTHONPATH="$OELLM_VLLM_SRC:$DEST:$PWD:$PWD/flash-linear-attention:$R/pylibs_flag:$R/pylibs_hfx"
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export HF_HOME=${FLAG_HF_HOME:-/gpfs/projects/ehpc533/users/korbip/data/flag_evals/hf_home}
export HF_HUB_CACHE=$HF_HOME/hub HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_METRICS_CACHE=$HF_HOME/metrics/job_${SLURM_JOB_ID:-local}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1 NLTK_DATA=$R/nltk_data TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OELLM_CPU_THREADS:-8}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARNING}
