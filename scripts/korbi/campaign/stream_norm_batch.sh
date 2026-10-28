#!/bin/bash
# stream_norm.py over several checkpoints in one pool job (inside the training container, run by
# config/experiments/korbi/hybrid_speed/kernel_tests_mn5.yaml's run_tests.sh from the repo root;
# user 2026-10-28: experiments on the cns144 pool host, not direct sbatch):
#
#   stream_norm_batch.sh <out dir> <datamix> <tag>|<train script>|<load dir>|<step> ... | @<spec file>
#
# A spec file holds one <tag>|<train script>|<load dir>|<step> per line (keeps the pipes out of
# the Hydra override that carries the command).
# One torchrun (4 GPUs) per checkpoint; <out dir>/<tag>.json and <tag>.log each. A failed one is
# reported and the rest still run.
set -u
OUT=${1:?out dir}
MIX=${2:?datamix}
shift 2
if [ "${1:-}" != "${1#@}" ]; then
    mapfile -t specs < <(grep -v '^\s*\(#\|$\)' "${1#@}")
    set -- "${specs[@]}"
fi
H=${HYBRID_ENV:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
export ANGULAR_MUOWN_COMPACT_CKPT=1 HF_HOME=$H/cache/hf HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/gpfs/scratch/ehpc533/users/korbip/tmp/triton_cache}
mkdir -p "$OUT"
status=0
for spec in "$@"; do
    IFS='|' read -r tag script load step <<< "$spec"
    python -m torch.distributed.run --standalone --nproc-per-node 4 \
        scripts/korbi/campaign/stream_norm.py --train-script "$script" --load "$load" \
        --ckpt-step "$step" --datamix "$MIX" --out "$OUT/$tag.json" --ep 4 \
        > "$OUT/$tag.log" 2>&1 || status=1
    echo "[stream_norm_batch] $tag: $([ -s "$OUT/$tag.json" ] && echo ok || echo FAILED)"
done
exit $status
