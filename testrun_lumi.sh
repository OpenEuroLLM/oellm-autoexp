module load cray-python/3.11.7
source .venv/bin/activate
source set_env.sh
export PYTHONPATH=$PYTHONPATH:$pwd
# experiment=experiments/megatron_lumi_speed_test.yaml
# experiment=experiments/megatron_dense_scaling_law_lumi
experiment=experiments/9B_sparse_sanity_check.yaml
OELLM_MEGATRON_SCHEMA_ONLY=1 python3 scripts/run_autoexp_container.py --config-ref $experiment $@