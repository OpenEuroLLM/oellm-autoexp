#!/bin/bash
# ckda_branch checkpoint surgery for the post-fit arms (add_ckda_branch.py apply): one checkpoint per arm,
# <ROOT>/<arm>_src/ckpt/iter_<IT> = the source's shards hard-linked + the branch tensors, and its
# latest_checkpointed_iteration.txt. An arm whose checkpoint exists is skipped. Run from a synced snapshot.
#
# Run as the 1-node job on p264 host003, config/experiments/korbi/production/ckda_branch_surgery_mn5.yaml
# (aux.mode=test: hybrid_v4_ckda_branch_test_mn5's checkpoints; aux.mode=prod: hybrid_ambitious_v4_ckda_branch_mn5's).
# Each output holds ckda_branch_surgery.json; add_ckda_branch.py verify --dst <ckpt> checks it reproduces.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SIF=${SIF:-/gpfs/projects/ehpc533/users/bsc088649/hybrid_v019/container/MegatronTraining-hybrid-ep_x86_64_202610161948.sif}
MODE=${1:?test or prod}
shift
case $MODE in
  test)
    SRC=${SRC:-/gpfs/scratch/ehpc533/users/korbip/tmp/v4hd256pool/fctrl/ckpt}
    ROOT=${ROOT:-/gpfs/scratch/ehpc533/users/korbip/tmp/ckdabranch}
    IT=${IT:-1100}
    ;;
  prod)
    SRC=${SRC:-/gpfs/scratch/ehpc533/hybrid_nucleus/production/main_mhc4/ambitious_e360_eng80_v4_2ph_hd256/ckpt_eval}
    ROOT=${ROOT:-/gpfs/scratch/ehpc533/hybrid_nucleus/production/ckda_branch}
    IT=${IT:-23300}
    ;;
  *) echo "mode: test or prod" >&2; exit 2 ;;
esac
# inside a container (a job's, e.g. kernel_tests_mn5 aux.bench) python directly, else through the image
if [ -n "${SINGULARITY_CONTAINER:-}${APPTAINER_CONTAINER:-}" ]; then PY=(python3); else PY=(singularity exec "$SIF" python3); fi
ARMS=("$@")
[ ${#ARMS[@]} -gt 0 ] || ARMS=(bgqa b8 cgqa d)
ITER=iter_$(printf %07d "$IT")
[ -d "$SRC/$ITER" ] || { echo "no $SRC/$ITER" >&2; exit 1; }

for arm in "${ARMS[@]}"; do
  case $arm in
    bgqa) args=(--branch attention --groups 2 --layers 9,25,41) ;;
    b8)   args=(--branch attention --groups 8 --layers 9,25,41) ;;
    cgqa) args=(--branch attention --groups 2 --layers 9,25,41,58,74,90) ;;
    d)    args=(--branch kronlstm --layers 9,25,41) ;;
    # attention, magnitude-matched start: GAMMA_INIT_BGQA = layer:(rms_ckda / rms_branch - 1),... (measured)
    bgqa_mm) args=(--branch attention --groups 2 --layers 9,25,41 --gamma-init "${GAMMA_INIT_BGQA:?layer:value,...}") ;;
    # KronLSTM, magnitude-matched start: GAMMA_INIT_D = layer:(rms_ckda / rms_branch - 1),... (measured)
    d_mm) args=(--branch kronlstm --layers 9,25,41 --gamma-init "${GAMMA_INIT_D:?layer:value,...}") ;;
    # the drop control: an attention branch at weight 0 (crossfade 1 + gamma = 0): complex_kda just fades out
    drop) args=(--branch attention --groups 2 --layers 9,25,41 --gamma-init -1) ;;
    # calibrated (ckda_branch_calibrate: the branch scaled to complex_kda's output RMS on the first step)
    bgqa_cal) args=(--branch attention --groups 2 --layers 9,25,41 --calibrate) ;;
    b8_cal)   args=(--branch attention --groups 8 --layers 9,25,41 --calibrate) ;;
    cgqa_cal) args=(--branch attention --groups 2 --layers 9,25,41,58,74,90 --calibrate) ;;
    d_cal)    args=(--branch kronlstm --layers 9,25,41 --calibrate) ;;
    *) echo "unknown arm $arm" >&2; exit 2 ;;
  esac
  dst=$ROOT/${arm}_src/ckpt
  if [ -f "$dst/latest_checkpointed_iteration.txt" ]; then
    echo "$arm: $dst exists, skipped"
    continue
  fi
  mkdir -p "$dst"
  rm -rf "${dst:?}/$ITER"  # a partial earlier attempt (no latest file yet)
  echo "$arm: ${args[*]} -> $dst/$ITER"
  CUDA_VISIBLE_DEVICES= "${PY[@]}" "$HERE/add_ckda_branch.py" apply \
    --src "$SRC/$ITER" --dst "$dst/$ITER" "${args[@]}"
  echo "$IT" > "$dst/latest_checkpointed_iteration.txt"
done
