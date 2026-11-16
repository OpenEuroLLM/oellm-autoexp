#!/bin/bash
# Add Megatron's attention_head_temperature parameters to the K surgery checkpoint (docs/recall_surgery_ambitious.md
# §20; the DK1T arm). The source and destination are on the same file system, so dcp_surgery.write_surgery
# hard-links the 1.1 TB of source .distcp files; only the new head-temperature tensors and the merged .metadata
# are written. Deterministic, no GPU.
#
#   SRC=<surgery>/DK1/ckpt/iter_0035900 DST=<surgery>/DK1T/ckpt/iter_0035900 IT=35900 \
#       bash scripts/korbi/recall_surgery/headtemp_to_dk1t.sh
set -euo pipefail
SRC=${SRC:?source checkpoint}
DST=${DST:?destination checkpoint}
IT=${IT:-35900}
DST_DIR=$(dirname "$DST")
# a partial earlier attempt (no tracker yet) is removed; the source is never touched
rm -rf "$DST_DIR"
python3 scripts/korbi/production/add_head_temperature.py --src "$SRC" --dst "$DST"
echo "$IT" > "$DST_DIR/latest_checkpointed_iteration.txt"
echo "[headtemp] $DST + tracker $IT"
