#!/bin/bash
#SBATCH --job-name=engram-branchgates-test
#SBATCH --account=ehpc533
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:40:00
#SBATCH --output=/gpfs/projects/ehpc533/users/rluukkon/oellm-autoexp-nucleus-engram/logs/%x-%j.out
# Per-stream Engram gates: unit tests on one GPU, then the checkpoint surgery as a dry run on the
# production checkpoint (read only). Everything is written under this workspace or node-local /tmp.
set -u
W=/gpfs/projects/ehpc533/users/rluukkon/oellm-autoexp-nucleus-engram
SIF=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/container/MegatronTraining-hybrid-ep_x86_64_202610161948.sif
H=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
LIBS=$H/pylibs_moonep_ec3eb8b:$H/retention:$H/pylibs_ep
CKPT=${CKPT:-/gpfs/scratch/ehpc533/hybrid_nucleus/production/v4_cont/runs/main/ckpt/iter_0039218}
T=/tmp/rluukkon_bg_$SLURM_JOB_ID
mkdir -p "$T"
cd "$W"
command -v singularity >/dev/null || module load singularity >/dev/null 2>&1 || true
run() {
    singularity exec --nv --bind /gpfs,/tmp \
        --env PYTHONPATH=$LIBS:.:submodules/Megatron-LM \
        --env PYTHONDONTWRITEBYTECODE=1 \
        --env TRITON_CACHE_DIR=$T/triton --env TORCHINDUCTOR_CACHE_DIR=$T/inductor \
        --env HOME=$T --env XDG_CACHE_HOME=$T/cache \
        "$SIF" "$@"
}
echo "== unit tests"
(cd submodules/Megatron-LM && run torchrun --standalone --nproc_per_node=1 -m pytest -p no:cacheprovider \
    -o addopts="" -W ignore -q -rfE \
    tests/unit_tests/ssm/test_engram_branch_gates.py tests/unit_tests/ssm/test_engram.py \
    tests/unit_tests/ssm/test_engram_output_scale.py ${EXTRA_TESTS:-} 2>&1 | tail -60)
echo "== memory / time at 64k tokens"
run torchrun --standalone --nproc_per_node=1 scripts/rluukkon/engram_branch_mem.py 2>&1 | grep -v -i warn | tail -8
[ -n "${SKIP_SURGERY:-}" ] && exit 0
echo "== surgery selftest"
run python3 scripts/korbi/production/add_engram_branch_gates.py selftest 2>&1 | tail -3
echo "== surgery dry run on $CKPT"
run python3 scripts/korbi/production/add_engram_branch_gates.py --src "$CKPT" --dst "$T/unused" \
    --query-mix --gate-scale --dry-run 2>&1 | tail -60
