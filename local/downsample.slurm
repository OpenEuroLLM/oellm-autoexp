#!/bin/bash
#SBATCH --job-name=iodata
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --partition=small
#SBATCH --time=04:00:00
#SBATCH --account=project_462000963
#SBATCH --output logs/%j.out
#SBATCH --error logs/%j.err

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 INDIR OUTFILE" >&2
    exit 1
fi


export CC=gcc-12
export CXX=g++-12

INFILE="$1"
OUTFILE="$2"

mkdir -p $(dirname "$OUTFILE")
echo "Downsampling $INFILE to $OUTFILE"
CONTAINER=/scratch/project_462000394/containers/for-turkunlp-team/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1-te-dockerhash-560b76ceab56.sif
export SINGULARITY_BIND=/pfs,/scratch,/projappl,/project,/flash,/appl,/usr/lib64/libjansson.so.4,/usr/lib64/libcxi.so.1,/opt/cray,/var/spool/slurmd
singularity exec --bind "$SINGULARITY_BIND" "$CONTAINER" \
    python3 /pfs/lustrep4/scratch/project_462000963/users/rluukkon/git/oellm-autoexp/local/downsample_indexed_dataset.py \
        --path $INFILE \
        --ratio 0.15 \
        --out-prefix $OUTFILE \
        --exit-after-iteration 10000