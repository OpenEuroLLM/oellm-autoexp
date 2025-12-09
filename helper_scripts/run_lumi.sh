module load cray-python/3.11.7
source .venv/bin/activate
source set_env.sh
export PYTHONPATH=$PYTHONPATH:$pwd
export WANDB_ENTITY=openeurollm-project
python3 scripts/run_autoexp_container.py --config-ref experiments/megatron_dense_scaling_law_lumi.yaml $@
