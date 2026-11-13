#!/bin/bash
# Early 3:1 surgery (docs/recall_surgery_ambitious.md §13; config/experiments/korbi/recall_surgery/early31_mn5.yaml)
# from the ambitious v4_2ph main run's ckpt_eval/iter_<IT> (head dim 128, 7:1; read only):
#   src/w256    widen_attention_head_dim.py apply   (6 attention sublayers 128 -> 256, function-preserving)
#   src/w256rk  widen_attention_head_dim.py reseed-k (new key channels = an orthogonal mix of the old: q_new gets
#               O(1) gradients; exact)
#   src/a31     replace_ckda_with_attention.py apply on w256rk: complex_kda 9/25/41/58/74/90 -> random attention
#               (row statistics of the unwidened checkpoint), the 3:1 pattern in the saved args
# then the arms' training directories on TRAIN_ROOT (scratch; iter_<IT> a symlink, tracker IT = the jobs' gate):
#   ctrl/ckpt/iter_<IT> -> ROOT/src/w256rk     a31/ckpt/iter_<IT> -> ROOT/src/a31
# Each step skips an output that has its tracker; a partial output (no tracker) is removed and redone. Login node
# (every step a separate process, < 300 s CPU each; hard links + ~1-3 GB read) or a 1-node job, from a synced snapshot.
#   IT=1000 early31_surgery.sh [w256 w256rk a31 seed]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# the saved args unpickle into megatron classes: the snapshot's Megatron on the path
REPO=$(cd "$HERE/../../.." && pwd)
export PYTHONPATH="$REPO/submodules/Megatron-LM:$REPO${PYTHONPATH:+:$PYTHONPATH}"
# the image sets its own PYTHONPATH: SINGULARITYENV_ / APPTAINERENV_ override it inside the container
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH" APPTAINERENV_PYTHONPATH="$PYTHONPATH"
SIF=${SIF:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019/container/MegatronTraining-hybrid-ep_x86_64_202610161948.sif}
SRC=${SRC:-/gpfs/scratch/ehpc533/hybrid_nucleus/production/main_mhc4/ambitious_e360_eng80_v4_2ph/ckpt_eval}
# on /gpfs/projects: the source checkpoint lives there (main_mhc4/... on scratch is a symlink), hard links need
# the same filesystem
ROOT=${ROOT:-/gpfs/projects/ehpc533/hybrid_nucleus/production/early31}
# the arms' training directories (load = save; all saves): scratch
TRAIN_ROOT=${TRAIN_ROOT:-/gpfs/scratch/ehpc533/hybrid_nucleus/production/early31}
IT=${IT:-1000}
LAYERS=9,25,41,58,74,90
REFS=0,17,33,50,66,82
PATTERN31='*ENLELELE*ELELELE*ELELELE*ELELELE*ELELELE*ELELELEN*ELELELE*ELELELE*ELELELE*ELELELE*ELELELE*ELELELE'
case $ROOT in */main_mhc4*) echo "refusing to write below the main run ($ROOT)" >&2; exit 2 ;; esac
if [ -n "${SINGULARITY_CONTAINER:-}${APPTAINER_CONTAINER:-}" ]; then PY=(python3); else
  command -v singularity >/dev/null || module load singularity >/dev/null 2>&1 || true
  PY=(singularity exec "$SIF" python3)
fi
ITER=iter_$(printf %07d "$IT")
steps=${*:-w256 w256rk a31 seed}

done_at() { [ -f "$1/latest_checkpointed_iteration.txt" ] && [ "$(tr -d '[:space:]' < "$1/latest_checkpointed_iteration.txt")" = "$IT" ]; }
fresh() { [ -e "$1/$ITER" ] && { echo "removing the partial $1/$ITER"; rm -rf "$1/$ITER"; }; mkdir -p "$1"; }
finish() { echo "$IT" > "$1/latest_checkpointed_iteration.txt"; echo "[$2] tracker $IT ($(date +%T))"; }

for s in $steps; do
  case $s in
    w256)
      O=$ROOT/src/w256/ckpt; done_at "$O" && { echo "[w256] done"; continue; }; fresh "$O"
      "${PY[@]}" "$HERE/widen_attention_head_dim.py" apply --src "$SRC/$ITER" --dst "$O/$ITER" --layers "$REFS" \
          --d-old 128 --d-new 256 --eps 1e-3 --seed 1234
      finish "$O" w256 ;;
    w256rk)
      O=$ROOT/src/w256rk/ckpt; done_at "$O" && { echo "[w256rk] done"; continue; }; fresh "$O"
      done_at "$ROOT/src/w256/ckpt" || { echo "[w256rk] needs w256" >&2; exit 3; }
      "${PY[@]}" "$HERE/widen_attention_head_dim.py" reseed-k --src "$ROOT/src/w256/ckpt/$ITER" --dst "$O/$ITER" \
          --layers "$REFS" --d-old 128 --d-new 256 --seed 1234
      finish "$O" w256rk ;;
    a31)
      O=$ROOT/src/a31/ckpt; done_at "$O" && { echo "[a31] done"; continue; }; fresh "$O"
      done_at "$ROOT/src/w256rk/ckpt" || { echo "[a31] needs w256rk" >&2; exit 3; }
      "${PY[@]}" "$HERE/replace_ckda_with_attention.py" apply --src "$ROOT/src/w256rk/ckpt/$ITER" --dst "$O/$ITER" \
          --stats-src "$SRC/$ITER" --layers "$LAYERS" --refs "$REFS" --seed 1234 --pattern "$PATTERN31"
      finish "$O" a31 ;;
    seed)
      # the training directories on SCRATCH (lead 11-13: projects nearly full): iter_<IT> a symlink to the src
      # output on projects (no copy; cross-filesystem hard links fail), tracker IT = the jobs' gate
      for pair in ctrl:w256rk a31:a31; do
        arm=${pair%%:*}; src=$ROOT/src/${pair#*:}/ckpt
        O=$TRAIN_ROOT/$arm/ckpt; done_at "$O" && { echo "[seed $arm] done"; continue; }
        if [ -f "$O/latest_checkpointed_iteration.txt" ]; then echo "[seed $arm] $O has another tracker" >&2; exit 3; fi
        done_at "$src" || { echo "[seed $arm] needs $src" >&2; exit 3; }
        mkdir -p "$O"; [ -L "$O/$ITER" ] && rm "$O/$ITER"
        [ -e "$O/$ITER" ] && { echo "[seed $arm] $O/$ITER exists (not a symlink)" >&2; exit 3; }
        ln -s "$src/$ITER" "$O/$ITER"
        [ -f "$O/$ITER/.metadata" ] || { echo "[seed $arm] $O/$ITER/.metadata unreadable" >&2; exit 3; }
        finish "$O" "seed $arm"
      done
      ;;
    *) echo "unknown step $s" >&2; exit 2 ;;
  esac
done
