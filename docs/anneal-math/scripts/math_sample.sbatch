#!/bin/bash
#SBATCH --job-name=math_sample
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --threads-per-core=1
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --account=e-sta-openeurollm
#SBATCH --output=/e/scratch/e-sta-openeurollm/poeppel1/dump/math_sample/logs/%x-%j.out
# Sampled AIME24/25, AMC23, MATH500 for one checkpoint: greedy prompt check, then 4 shards (one per GPU).
#   sbatch --job-name=ms_<ck> math_sample.sbatch <ck>
CK=$1
D=/e/scratch/e-sta-openeurollm/poeppel1/dump/math_sample
V=/e/scratch/e-sta-openeurollm/poeppel1/flag-evals/views/identity/$CK
SIF=/e/project1/e-sta-openeurollm/container/oellm-eval-vllm.sif
# Own writable home per process (FlashInfer/vLLM caches); the real /e/home is not mounted in the container.
run() {
    H=$D/home/${SLURM_JOB_ID}_$1; mkdir -p $H
    singularity exec --nv --cleanenv --containall --no-mount bind-paths,hostfs,cwd,home --home $H \
        --env TMPDIR=/tmp --env TMP=/tmp --env TEMP=/tmp \
        --env HF_HUB_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
        --env RAY_USAGE_STATS_ENABLED=0 --env OMP_NUM_THREADS=8 --env VLLM_ENABLE_V1_MULTIPROCESSING=0 \
        --bind /e/scratch/e-sta-openeurollm:/e/scratch/e-sta-openeurollm \
        $SIF env CUDA_VISIBLE_DEVICES=$1 python3 $D/math_sample.py --view $V --name $CK --out $D/out "${@:2}"
}
echo "start $(date) $CK on $(hostname)"
run 0 --check > $D/logs/$CK.check.log 2>&1 &
for i in 1 2 3; do run $i --shard $i --nshards 4 > $D/logs/$CK.shard$i.log 2>&1 & done
wait %1
echo "check done $(date)"
run 0 --shard 0 --nshards 4 > $D/logs/$CK.shard0.log 2>&1
wait
echo "end $(date)"; ls $D/out/$CK.*
