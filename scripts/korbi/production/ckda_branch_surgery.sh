#!/bin/bash
# ckda_branch checkpoint surgery for the post-fit arms (add_ckda_branch.py apply): one checkpoint per arm,
# <ROOT>/<arm>_src/ckpt/iter_<IT> = the source's shards hard-linked + the branch tensors, and its
# latest_checkpointed_iteration.txt. An arm whose checkpoint exists is skipped. Run from a synced snapshot.
#
#   test (hybrid_v4_ckda_branch_test_mn5; small model, a login node is fine):
#     scripts/korbi/production/ckda_branch_surgery.sh test
#   production (hybrid_ambitious_v4_ckda_branch_mn5; the 84B checkpoint's .metadata: run in a job):
#     scripts/korbi/mn5_autoexp_bg.sh ckdab_surgery scripts/run_autoexp.py \
#         --config-name experiments/korbi/hybrid_speed/kernel_tests_mn5 --yes aux.name=ckda_branch_surgery \
#         'aux.bench="bash scripts/korbi/production/ckda_branch_surgery.sh prod"' slurm.sbatch.time=02:00:00
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
