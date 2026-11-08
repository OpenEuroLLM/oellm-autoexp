#!/bin/bash
# Replace the model code of oellm_ambitious HF exports (or their views) with this checkout's,
# atomically: each file is copied next to its target and renamed over it (a new inode), so hard
# links between exports and views are never written through and a running process keeps the
# files it opened. Weights, config.json and the tokenizer stay untouched. Exports without
# modeling_oellm_ambitious.py (q35, oellm_hybrid) are skipped.
#
#   scripts/korbi/hf_export/refresh_code.sh <export dir> [<export dir> ...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HF=$ROOT/oellm_autoexp/hf_export
files=("$HF"/oellm_ambitious/configuration_oellm_ambitious.py "$HF"/oellm_ambitious/modeling_oellm_ambitious.py
       "$HF"/oellm_ambitious/mhc_kernels.py "$HF"/oellm_ambitious/oellm_kernels.py
       "$HF"/oellm_hybrid/configuration_oellm_hybrid.py "$HF"/oellm_hybrid/modeling_oellm_hybrid.py)
for d in "$@"; do
    if [ ! -f "$d/modeling_oellm_ambitious.py" ]; then
        echo "[refresh_code] skip (not an oellm_ambitious export): $d"
        continue
    fi
    changed=0
    for f in "${files[@]}"; do
        dst="$d/$(basename "$f")"
        if [ -f "$dst" ] && cmp -s "$f" "$dst"; then
            continue
        fi
        tmp="$dst.new.$$"
        cp "$f" "$tmp" && mv -f "$tmp" "$dst"
        changed=$((changed + 1))
    done
    echo "[refresh_code] $d: $changed file(s) replaced"
done
