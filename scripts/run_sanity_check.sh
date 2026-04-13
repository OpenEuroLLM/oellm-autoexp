#!/bin/bash
# Cross-cluster Megatron sanity check runner.
#
# Usage: bash run_sanity_check.sh [extra hydra overrides...]
#   e.g. bash run_sanity_check.sh slurm.sbatch.nodes=4
#
# Prerequisites:
#   1. Download data once per cluster:
#        mkdir -p "$DATA_DIR/gpt-neox-20b/common-pile"
#        wget -P "$DATA_DIR/gpt-neox-20b/common-pile" \
#            https://462000963.lumidata.eu/common-pile/wikipesto.bin \
#            https://462000963.lumidata.eu/common-pile/wikipesto.idx \
#            https://462000963.lumidata.eu/common-pile/wikipesto.info.json

# Auto-detect cluster, partition and account from hostname (env vars take precedence).
if [[ -z "${CLUSTER:-}" ]]; then
    { read -r CLUSTER; read -r SLURM_PARTITION; read -r SLURM_ACCOUNT; } \
        < <(uv run --python 3.12 python scripts/detect_cluster.py --fields)
    export CLUSTER SLURM_PARTITION SLURM_ACCOUNT
fi


echo "Running sanity check on cluster: $CLUSTER"
# switch case for DATA_DIR based on CLUSTER

case $CLUSTER in
    lumi)
        export DATA_DIR="/pfs/lustrep4/scratch/project_462000963/preprocessed"
        ;;
    jupiter)
        echo "TODO: Not defined yet"
        exit 1
        ;;
    leonardo)
        echo "TODO: Not defined yet"
        exit 1
        ;;
    juwels)
        echo "TODO: Not defined yet"
        exit 1
        ;;
    mi325x)
        echo "TODO: Not defined yet"
        exit 1
        ;;
    *)
        echo "TODO: Not defined yet"
        exit 1
        ;;
esac


uv run --python 3.12 python scripts/run_autoexp.py \
    --config-name experiments/megatron_sanity_checks/megatron_sanity_check \
    container=$CLUSTER slurm=$CLUSTER \
    "$@"
