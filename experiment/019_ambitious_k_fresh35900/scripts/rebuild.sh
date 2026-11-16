#!/bin/bash
# Fresh target only; all original checkpoint files remain read-only.
set -euo pipefail
source_iter=${SRC:?}/iter_0035900
target_iter=${ROOT:?}/DK1/ckpt/iter_0035900
test ! -e "$target_iter"
test "$(tr -d '[:space:]' < "$SRC/latest_checkpointed_iteration.txt")" = 35900
mkdir -p "$ROOT"
sha256sum "$source_iter/.metadata" > "$ROOT/source_metadata.before.sha256"
bash scripts/korbi/recall_surgery/ds84_surgery.sh dk6
sha256sum --check "$ROOT/source_metadata.before.sha256"
echo '[fresh35900] surgery verified; original metadata unchanged'
