module load cray-python/3.11.7
source .venv/bin/activate
source set_env.sh
export PYTHONPATH=$PYTHONPATH:$pwd


OELLM_MEGATRON_SCHEMA_ONLY=1 python3 scripts/run_autoexp_container.py --config-ref experiments/rluukkon/moe_0.7B_115M_dense_input_output.yaml --no-monitor
OELLM_MEGATRON_SCHEMA_ONLY=1 python3 scripts/run_autoexp_container.py --config-ref experiments/rluukkon/moe_0.7B_115M_shared_expert.yaml --no-monitor
OELLM_MEGATRON_SCHEMA_ONLY=1 python3 scripts/run_autoexp_container.py --config-ref experiments/rluukkon/moe_0.7B_115M.yaml --no-monitor