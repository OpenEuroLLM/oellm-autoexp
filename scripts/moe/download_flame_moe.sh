#!/usr/bin/env bash
# Download the CMU-FLAME FLAME-MoE model family (Megatron torch_dist checkpoints)
# into a predictable local dir so the offline compute nodes can --load them.
#
# Each HF repo IS a Megatron checkpoint dir: it contains latest_checkpointed_iteration.txt
# plus one or more iter_<step>/ dirs of .distcp shards. We download with --local-dir so
# the path is clean (no snapshot hash), which is what compute_moe_metrics.py --load wants.
#
# Run this on a LOGIN node (compute nodes have no internet).
#
# Usage:
#   bash scripts/moe/download_flame_moe.sh                          # all 7 models
#   MODELS="FLAME-MoE-38M-100M" bash scripts/moe/download_flame_moe.sh
#   FLAME_ROOT=/some/other/path bash scripts/moe/download_flame_moe.sh
set -euo pipefail

# Checkpoints are large (the 1.7B-10.3B repo has 10 iters); keep them on scratch.
FLAME_ROOT="${FLAME_ROOT:-/leonardo_scratch/large/userexternal/ajha0001/flame_moe}"
mkdir -p "$FLAME_ROOT"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
HF_DL_MAX_WORKERS="${HF_DL_MAX_WORKERS:-2}"

# Full family. The trailing comment is the released checkpoint count (FYI only).
MODELS="${MODELS:-\
FLAME-MoE-38M-100M \
FLAME-MoE-98M-349M \
FLAME-MoE-115M-459M \
FLAME-MoE-290M-1.3B \
FLAME-MoE-419M-2.2B \
FLAME-MoE-721M-3.8B \
FLAME-MoE-1.7B-10.3B}"

# Locate the HF CLI (new `hf`, fall back to `huggingface-cli`).
if command -v hf >/dev/null 2>&1; then
  HF_CLI="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI="huggingface-cli"
else
  echo "neither 'hf' nor 'huggingface-cli' found; installing huggingface_hub[cli]..."
  pip install --quiet --user "huggingface_hub[cli]>=0.34"
  export PATH="$HOME/.local/bin:$PATH"
  command -v hf >/dev/null 2>&1 && HF_CLI="hf" || HF_CLI="huggingface-cli"
fi
echo "Using HF CLI: $HF_CLI"
echo "FLAME_ROOT=$FLAME_ROOT"

flame_download () {
  local name="$1"
  local repo="CMU-FLAME/${name}"
  local dest="${FLAME_ROOT}/${name}"
  echo "================================================================"
  echo "Downloading: ${repo}  ->  ${dest}"
  echo "================================================================"
  mkdir -p "$dest"
  if [[ "$HF_CLI" == "hf" ]]; then
    hf download "$repo" --repo-type model --local-dir "$dest" \
      --max-workers "$HF_DL_MAX_WORKERS"
  else
    huggingface-cli download "$repo" --repo-type model --local-dir "$dest" \
      --resume-download --max-workers "$HF_DL_MAX_WORKERS"
  fi
  # Sanity: a valid Megatron load dir has the tracker file + at least one iter_ dir.
  if [[ -f "${dest}/latest_checkpointed_iteration.txt" ]]; then
    echo "  latest_checkpointed_iteration: $(cat "${dest}/latest_checkpointed_iteration.txt")"
    echo "  iter dirs: $(find "$dest" -maxdepth 1 -type d -name 'iter_*' | wc -l)"
  else
    echo "  WARNING: no latest_checkpointed_iteration.txt in ${dest}" >&2
  fi
}

for m in $MODELS; do
  flame_download "$m"
done

# FLAME's training tokenizer (EleutherAI/pythia-12b). Tokenizer files only — no model
# weights — so this is tiny. Used at Megatron init via --tokenizer-type HuggingFaceTokenizer.
PYTHIA_TOK_DIR="${PYTHIA_TOK_DIR:-/leonardo_scratch/large/userexternal/ajha0001/tokenizers/pythia-12b}"
echo "================================================================"
echo "Downloading tokenizer: EleutherAI/pythia-12b  ->  ${PYTHIA_TOK_DIR}"
echo "================================================================"
mkdir -p "$PYTHIA_TOK_DIR"
TOK_PATTERNS=(tokenizer.json tokenizer_config.json special_tokens_map.json)
if [[ "$HF_CLI" == "hf" ]]; then
  hf download EleutherAI/pythia-12b --repo-type model --local-dir "$PYTHIA_TOK_DIR" \
    "${TOK_PATTERNS[@]/#/--include=}"
else
  huggingface-cli download EleutherAI/pythia-12b --repo-type model --local-dir "$PYTHIA_TOK_DIR" \
    --resume-download "${TOK_PATTERNS[@]/#/--include=}"
fi
if [[ -f "${PYTHIA_TOK_DIR}/tokenizer.json" ]]; then
  echo "  pythia-12b tokenizer.json present."
else
  echo "  WARNING: pythia-12b tokenizer.json missing in ${PYTHIA_TOK_DIR}" >&2
fi

echo
echo "Done. Disk usage per model:"
du -sh "$FLAME_ROOT"/* 2>/dev/null || true
