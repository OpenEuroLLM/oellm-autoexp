#!/usr/bin/env bash
# Download released MoE models into the scratch HF cache so SLURM jobs don't
# pull 30-60 GB over the wire on each launch. Idempotent: if a snapshot is
# already in the cache, huggingface-cli is a no-op.
#
# Usage:
#   bash scripts/moe/download_hf_models.sh                  # all default models
#   MODELS="Qwen/Qwen3-30B-A3B" bash scripts/moe/download_hf_models.sh
#   HF_HOME=/some/other/path bash scripts/moe/download_hf_models.sh
#
# For OLMoE we ALSO fetch a representative set of intermediate-step revisions
# (the model publishes hundreds; pass OLMOE_REVISIONS="" to skip).
set -euo pipefail

# Default cache root: scratch, not work — these blobs are large.
export HF_HOME="${HF_HOME:-/leonardo_scratch/large/userexternal/ajha0001/HF_CACHE}"
mkdir -p "$HF_HOME"

# Stay offline-friendly post-download.
# hf_transfer + many workers OOMs the Leonardo login node on big shards
# (Step-3.5-Flash, Qwen3-30B). Default: disabled, single worker. Override
# both env vars if you're running this on a fat machine.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
HF_DL_MAX_WORKERS="${HF_DL_MAX_WORKERS:-2}"

MODELS="${MODELS:-Qwen/Qwen3-30B-A3B stepfun-ai/Step-3.5-Flash allenai/OLMoE-1B-7B-0924}"

# OLMoE publishes ~244 step-branches. Compute nodes on Leonardo have no
# internet, so EVERY revision the submit driver will use must be cached
# locally first. Defaults to every-10th step + main, matching
# `scripts/moe/submit_hf_metrics.sh` (OLMOE_EVERY=10).
#   OLMOE_EVERY=20 ...      -> coarser cadence
#   OLMOE_EVERY=1 ...       -> ALL ~244 revisions (huge: ~2.5 TB)
#   OLMOE_REVISIONS="..."   -> explicit override; skips list_olmoe_revisions.py
#   OLMOE_REVISIONS=""      -> skip OLMoE revision downloads entirely
OLMOE_EVERY="${OLMOE_EVERY:-10}"

if [[ -z "${OLMOE_REVISIONS+x}" ]]; then
  echo "Listing OLMoE revisions (every ${OLMOE_EVERY}th + main)..."
  OLMOE_REVISIONS="$(python3 scripts/moe/list_olmoe_revisions.py --every "${OLMOE_EVERY}")"
fi

echo "HF_HOME=$HF_HOME"
echo "Models: $MODELS"
if [[ -n "$OLMOE_REVISIONS" ]]; then
  n_rev=$(echo "$OLMOE_REVISIONS" | wc -w)
  echo "OLMoE revisions to fetch: $n_rev"
fi

# Locate the HF CLI. huggingface_hub>=0.34 ships `hf`; older versions ship
# `huggingface-cli`. Prefer the new name, fall back to the old, and install
# if neither is present.
if command -v hf >/dev/null 2>&1; then
  HF_CLI="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI="huggingface-cli"
else
  echo "neither 'hf' nor 'huggingface-cli' found; installing huggingface_hub[cli]..."
  pip install --quiet --user "huggingface_hub[cli]>=0.34"
  export PATH="$HOME/.local/bin:$PATH"
  if command -v hf >/dev/null 2>&1; then
    HF_CLI="hf"
  else
    HF_CLI="huggingface-cli"
  fi
fi
echo "Using HF CLI: $HF_CLI"

# Sub-command differs: new `hf download <repo>`, old `huggingface-cli download <repo>`.
# `--max-workers` was renamed; `--resume-download` is now the default. Build
# the arg vector once so the loop body stays readable.
hf_download () {
  local repo="$1"
  local revision="${2:-}"
  if [[ "$HF_CLI" == "hf" ]]; then
    if [[ -n "$revision" ]]; then
      hf download "$repo" --revision "$revision" --repo-type model \
        --max-workers "$HF_DL_MAX_WORKERS"
    else
      hf download "$repo" --repo-type model \
        --max-workers "$HF_DL_MAX_WORKERS"
    fi
  else
    if [[ -n "$revision" ]]; then
      huggingface-cli download "$repo" --revision "$revision" \
        --repo-type model --resume-download --max-workers "$HF_DL_MAX_WORKERS"
    else
      huggingface-cli download "$repo" \
        --repo-type model --resume-download --max-workers "$HF_DL_MAX_WORKERS"
    fi
  fi
}

for model in $MODELS; do
  echo "================================================================"
  echo "Downloading: ${model}  (main revision)"
  echo "================================================================"
  hf_download "$model"

  if [[ "$model" == "allenai/OLMoE-1B-7B-0924" && -n "$OLMOE_REVISIONS" ]]; then
    for rev in $OLMOE_REVISIONS; do
      [[ "$rev" == "main" ]] && continue  # already fetched above
      echo "  + OLMoE revision: ${rev}"
      hf_download "$model" "$rev" || echo "    (revision ${rev} unavailable, skipping)"
    done
  fi
done

echo
echo "Done. Disk usage:"
du -sh "$HF_HOME"/hub/* 2>/dev/null || true
