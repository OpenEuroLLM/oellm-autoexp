#!/bin/bash
# v4b copy-init surgery (docs/v4b_plan.md, option A): the main run's 35900 save -> a loadable
# 104-layer checkpoint whose six V layers are copies of their source attention layers, with two
# learnable gates at 0.
#
#   <ROOT>/insert/ckpt/iter_<IT>   insert_v_layers.py --init copy   q / gate / o / q-norm copied from
#                                  source attention 0/18/35/53/70/87 (old 0/17/33/50/66/82), the
#                                  source's input-norm gain folded into linear_qkv, and
#                                  v_gate.out_gate / v_gate.res_gate written at 0 with Adam state
#   <ROOT>/mhc/ckpt/iter_<IT>      add_v_mhc.py                     the six mHC attention mixers
#   <ROOT>/ckpt/iter_<IT>          fix_v_args.py                    args: 104 layers + V pattern
#
# then verify_v_surgery.py --init copy. A stage whose tracker holds IT is skipped; a partial output
# (no tracker) is removed and redone. Run from a synced snapshot as the 1-node pool job
# config/experiments/korbi/recall_surgery/v4b_surgery_mn5.yaml.
#   ROOT=<root> IT=35900 v4b_surgery.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SIF=${SIF:-/gpfs/projects/ehpc533/users/bsc088649/hybrid_v019/container/MegatronTraining-hybrid-ep_x86_64_202610161948.sif}
SRC=${SRC:-/gpfs/scratch/ehpc533/hybrid_nucleus/production/main_mhc4/ambitious_e360_eng80_v4_2ph_hd256/ckpt_rolling}
ROOT=${ROOT:-/gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_k_fresh35900_20261116/runs/v4b_src}
IT=${IT:-35900}
SOURCES=${SOURCES:-0,18,35,53,70,87}
if [ -n "${SINGULARITY_CONTAINER:-}${APPTAINER_CONTAINER:-}" ]; then PY=(python3); else PY=(singularity exec "$SIF" python3); fi
ITER=iter_$(printf %07d "$IT")
[ -d "$SRC/$ITER" ] || { echo "no $SRC/$ITER" >&2; exit 1; }
case $ROOT in */main_mhc4*) echo "refusing to write below the main run ($ROOT)" >&2; exit 2 ;; esac

done_at() { [ -f "$1/latest_checkpointed_iteration.txt" ] && [ "$(tr -d '[:space:]' < "$1/latest_checkpointed_iteration.txt")" = "$IT" ]; }
fresh() { [ -e "$1/$ITER" ] && { echo "removing the partial $1/$ITER"; rm -rf "$1/$ITER"; }; mkdir -p "$1"; }

O=$ROOT/insert/ckpt
if done_at "$O"; then echo "[insert] done"; else
  fresh "$O"
  "${PY[@]}" "$HERE/insert_v_layers.py" --init copy --sources "$SOURCES" --src "$SRC/$ITER" --dst "$O/$ITER"
fi

O=$ROOT/mhc/ckpt
if done_at "$O"; then echo "[mhc] done"; else
  fresh "$O"
  "${PY[@]}" "$HERE/add_v_mhc.py" --src "$ROOT/insert/ckpt/$ITER" --dst "$O/$ITER"
fi

O=$ROOT/ckpt
if done_at "$O"; then echo "[args] done"; else
  fresh "$O"
  "${PY[@]}" "$HERE/fix_v_args.py" --src "$ROOT/mhc/ckpt/$ITER" --dst "$O/$ITER"
fi

"${PY[@]}" "$HERE/verify_v_surgery.py" --init copy --sources "$SOURCES" \
    --src "$SRC/$ITER" --dst "$ROOT/ckpt/$ITER" --sample-values
echo "v4b source ready: $ROOT/ckpt ($IT)"
