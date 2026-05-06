#!/bin/bash
# Stage dataset to per-node NVMe before training.
#
# Usage: bash scripts/stage_nvme.sh <src_data_args_path> <dst_data_args_path>
#
#   src_data_args_path : original data_args file on GPFS (weight prefix pairs)
#   dst_data_args_path : where to write the new data_args file (GPFS output dir)
#
# If the data fits in $TMPDIR (minus NVME_MARGIN_GB), copies all .bin/.idx to
# $TMPDIR and writes dst with NVMe paths. Otherwise writes dst with original
# GPFS paths so training falls back to GPFS transparently.

set -euo pipefail

SRC_DATA_ARGS="${1:-}"
DST_DATA_ARGS="${2:-}"
MARGIN_GB="${NVME_MARGIN_GB:-40}"

NODE="[$(hostname) stage_nvme]"

# ── Guard: file-based data path required ─────────────────────────────────────
if [[ -z "$SRC_DATA_ARGS" || ! -f "$SRC_DATA_ARGS" ]]; then
    echo "$NODE WARNING: no valid data_args_path file provided ('$SRC_DATA_ARGS')."
    echo "$NODE NVMe staging only works with --data-args-path (file-based data config)."
    echo "$NODE If using --data-path inline, skip the staging template."
    echo "$NODE Training will read from GPFS as normal."
    exit 0
fi

if [[ -z "$DST_DATA_ARGS" ]]; then
    echo "$NODE ERROR: dst_data_args_path argument is required."
    exit 1
fi

echo "$NODE Source data_args: $SRC_DATA_ARGS"
echo "$NODE Destination data_args: $DST_DATA_ARGS"

# ── Parse weight/prefix pairs (same logic as Megatron: just split()) ──────────
mapfile -d '' TOKENS < <(tr -s '[:space:]' '\0' < "$SRC_DATA_ARGS" | sed '/^$/d')
# Build parallel arrays: weights and gpfs prefixes
WEIGHTS=()
PREFIXES=()
i=0
for tok in "${TOKENS[@]}"; do
    if (( i % 2 == 0 )); then
        WEIGHTS+=("$tok")
    else
        PREFIXES+=("$tok")
    fi
    (( i++ )) || true
done

N=${#PREFIXES[@]}
echo "$NODE Found $N dataset(s) in $SRC_DATA_ARGS"

# ── Compute total size of all .bin + .idx files ───────────────────────────────
TOTAL_BYTES=0
for prefix in "${PREFIXES[@]}"; do
    for ext in bin idx; do
        f="${prefix}.${ext}"
        if [[ ! -f "$f" ]]; then
            echo "$NODE ERROR: expected file not found: $f"
            exit 1
        fi
        sz=$(stat -c%s "$f")
        TOTAL_BYTES=$(( TOTAL_BYTES + sz ))
    done
done
TOTAL_GB=$(( TOTAL_BYTES / 1024 / 1024 / 1024 ))
echo "$NODE Total data size: ${TOTAL_GB}GB"

# ── Check available space in TMPDIR ──────────────────────────────────────────
AVAIL_KB=$(df -k "$TMPDIR" | awk 'NR==2 {print $4}')
AVAIL_GB=$(( AVAIL_KB / 1024 / 1024 ))
USABLE_GB=$(( AVAIL_GB - MARGIN_GB ))
echo "$NODE TMPDIR: $TMPDIR  available: ${AVAIL_GB}GB  margin: ${MARGIN_GB}GB  usable: ${USABLE_GB}GB"

# ── Fallback: write GPFS paths if data doesn't fit ───────────────────────────
if (( TOTAL_GB > USABLE_GB )); then
    echo "$NODE WARNING: data (${TOTAL_GB}GB) exceeds usable NVMe (${USABLE_GB}GB)."
    echo "$NODE Falling back to GPFS paths — training will read from GPFS."
    mkdir -p "$(dirname "$DST_DATA_ARGS")"
    cp "$SRC_DATA_ARGS" "$DST_DATA_ARGS"
    echo "$NODE Wrote passthrough data_args: $DST_DATA_ARGS"
    exit 0
fi

# ── Stage: copy each .bin and .idx to TMPDIR ─────────────────────────────────
NVME_DATA_DIR="$TMPDIR/oellm_data"
mkdir -p "$NVME_DATA_DIR"
echo "$NODE Staging to $NVME_DATA_DIR ..."

NVME_PREFIXES=()
for prefix in "${PREFIXES[@]}"; do
    # Use last two path components to avoid collisions (e.g. hplt3.../spa_Latn)
    rel=$(basename "$(dirname "$prefix")")/$(basename "$prefix")
    nvme_prefix="$NVME_DATA_DIR/$rel"
    mkdir -p "$(dirname "$nvme_prefix")"

    t0=$(date +%s)
    cp "${prefix}.idx" "${nvme_prefix}.idx"
    cp "${prefix}.bin" "${nvme_prefix}.bin"
    t1=$(date +%s)
    sz=$(du -sh "${nvme_prefix}.bin" | cut -f1)
    echo "$NODE  copied $rel  (${sz}, $((t1-t0))s)"

    NVME_PREFIXES+=("$nvme_prefix")
done

# ── Write new data_args file with NVMe paths ──────────────────────────────────
mkdir -p "$(dirname "$DST_DATA_ARGS")"
> "$DST_DATA_ARGS"
for (( j=0; j<N; j++ )); do
    echo "${WEIGHTS[$j]} ${NVME_PREFIXES[$j]}" >> "$DST_DATA_ARGS"
done

echo "$NODE NVMe staging complete. New data_args: $DST_DATA_ARGS"
cat "$DST_DATA_ARGS"
