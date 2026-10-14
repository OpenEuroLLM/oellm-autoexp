#!/bin/bash
# Prepare JUPITER for the architecture-scaling downstream eval sweep.
#
#   bash scripts/setup_eval_env_jupiter.sh            # venv + HF_HOME layout
#   bash scripts/setup_eval_env_jupiter.sh --prefetch # + download the datasets
#
# Run this on a LOGIN node: compute nodes have no internet, so every dataset the
# eval touches has to be in the HF cache beforehand.
#
# What it does
#   1. Initialises the oellm-eval submodule.
#   2. Installs it into $OELLM_EVAL_VENV with --no-deps. Deliberately no-deps:
#      the venv already carries datasets 4.x together with the fla /
#      mlstm_kernels / mamba_ssm builds the converted models need, and nothing
#      may replace them. Its actual imports (pandas, jsonargparse, rich,
#      huggingface_hub, pyyaml, datasets) are all present already.
#   3. Builds $OELLM_EVAL_HF_HOME, whose `datasets` entry is a symlink onto the
#      real populated cache. oellm-eval's generated script hardcodes
#      `HF_DATASETS_CACHE=$HF_HOME/datasets`, whereas JUPITER's cache keeps the
#      `owner___name` dirs at the cache ROOT; the symlink reconciles the two
#      without copying or re-downloading the ~350 datasets already there.
set -euo pipefail

REPO="${REPO:-$HOME/work/Projects/oellm-autoexp-hybrid}"
VENV="${OELLM_EVAL_VENV:-$HOME/work/venv}"
REAL_HF_HOME="${HF_HOME:?set HF_HOME to the populated cache root}"
EVAL_HF_HOME="${OELLM_EVAL_HF_HOME:-${REAL_HF_HOME}_eval}"
PREFETCH=0
[[ "${1:-}" == "--prefetch" ]] && PREFETCH=1

echo "repo         : $REPO"
echo "venv         : $VENV"
echo "real HF_HOME : $REAL_HF_HOME"
echo "eval HF_HOME : $EVAL_HF_HOME"

# --- 1. submodule -----------------------------------------------------------
# JUPITER has no GitHub SSH key, so the submodule clone fails here; in that case
# the working tree is expected to have been rsync'ed over (sync_to_jupiter.sh).
cd "$REPO"
if [[ ! -f submodules/oellm-eval/pyproject.toml ]]; then
    git submodule update --init submodules/oellm-eval || true
fi
if [[ ! -f submodules/oellm-eval/pyproject.toml ]]; then
    echo "ERROR: submodules/oellm-eval is empty and could not be cloned." >&2
    echo "       rsync it from a machine with GitHub access, e.g." >&2
    echo "       rsync -a --delete submodules/oellm-eval/ jupiter:$REPO/submodules/oellm-eval/" >&2
    exit 1
fi
if [[ -e submodules/oellm-eval/.git ]]; then
    echo "oellm-eval   : $(git -C submodules/oellm-eval rev-parse --short HEAD)"
else
    echo "oellm-eval   : (rsynced working tree, no .git)"
fi

# --- 2. venv ----------------------------------------------------------------
# `$VENV/bin/pip` carries a stale shebang (this venv was relocated), so drive
# pip through the interpreter instead of the console script.
PYTHONPATH= "$VENV/bin/python" -m pip install --no-deps -e "$REPO/submodules/oellm-eval"
"$VENV/bin/python" -c "import oellm, oellm.main; print('oellm OK:', oellm.__file__)"
"$VENV/bin/oellm-eval" --help >/dev/null && echo "oellm-eval CLI OK"

# --- 2b. datasets version note ----------------------------------------------
# datasets must be 4.x: the Hub serves dataset-card metadata in the 4.x format,
# which 3.x cannot parse ("Feature type 'List' not found"), so under 3.x even
# plain datasets like ai2_arc fail to resolve. oellm-eval itself now requires
# datasets>=4.0. Of the 24 (dataset, subset) probes covering both suites, the one
# 4.x could not load was the `winograd_wsc` loading script behind wsc273;
# oellm-eval reads the parquet `lighteval/winograd_wsc` instead.
"$VENV/bin/python" -c "import datasets; print('datasets', datasets.__version__)"

# --- 3. HF cache layout -----------------------------------------------------
# `$EVAL_HF_HOME/datasets` is a real dir of symlinks into the shared cache, with
# the links dropped for any entry the pinned datasets version cannot read (the
# shared cache is a mix of 3.x- and 4.x-built dirs). Nothing in the shared cache
# is modified; dropped entries are re-downloaded into the eval cache below.
mkdir -p "$EVAL_HF_HOME/datasets" "$EVAL_HF_HOME/hub"
# facebook/flores is gated: the eval HF_HOME needs the account token too.
[[ -e "$EVAL_HF_HOME/token" || ! -e "$REAL_HF_HOME/token" ]] || ln -s "$REAL_HF_HOME/token" "$EVAL_HF_HOME/token"
"$VENV/bin/python" "$REPO/scripts/link_eval_dataset_cache.py" \
    --shared-cache "$REAL_HF_HOME" --eval-cache "$EVAL_HF_HOME/datasets"

# --- 4. dataset prefetch ----------------------------------------------------
if [[ "$PREFETCH" == "1" ]]; then
    echo "prefetching dclm-core-22 + oellm-multilingual (needs internet) ..."
    env -u HF_HUB_OFFLINE -u HF_DATASETS_OFFLINE \
        HF_HOME="$EVAL_HF_HOME" \
        HF_DATASETS_CACHE="$EVAL_HF_HOME/datasets" \
        "$VENV/bin/python" "$REPO/scripts/link_eval_dataset_cache.py" \
            --shared-cache "$REAL_HF_HOME" \
            --eval-cache "$EVAL_HF_HOME/datasets" \
            --download
fi

echo "done."
