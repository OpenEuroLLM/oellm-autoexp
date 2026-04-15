#!/bin/bash
# Usage: bash scripts/run_experiment.sh experiments/rluukkon/<path>.yaml [hydra overrides...]
# e.g.:  bash scripts/run_experiment.sh experiments/rluukkon/moe_scaling_laws/moe_2B_400M.yaml
#        bash scripts/run_experiment.sh experiments/rluukkon/moe_scaling_laws/moe_2B_400M.yaml slurm.sbatch.nodes=4

CONFIG_BASE="${1:?Usage: run_experiment.sh experiments/rluukkon/<path>.yaml [hydra overrides...]}"
CONFIG_BASE="${CONFIG_BASE%.yaml}"
shift

# Auto-detect cluster, partition and account from hostname (env vars take precedence).
if [[ -z "${CLUSTER:-}" ]]; then
    { read -r CLUSTER; read -r SLURM_PARTITION; read -r SLURM_ACCOUNT; } \
        < <(uv run --python 3.12 python scripts/detect_cluster.py --fields)
    export CLUSTER SLURM_PARTITION SLURM_ACCOUNT
fi

echo "Running ${CONFIG_BASE} on cluster: ${CLUSTER}"

uv run --python 3.12 python scripts/run_autoexp.py \
    --config-name "${CONFIG_BASE}" \
    container="${CLUSTER}" slurm="${CLUSTER}" "${CONFIG_BASE}=${CLUSTER}" \
    "$@"
