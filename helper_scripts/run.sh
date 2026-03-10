#!/bin/bash
module load cray-python/3.11.7
source /scratch/project_462000963/users/rluukkon/git/oellm-autoexp/.venv/bin/activate
source helper_scripts/set_env.sh
export PYTHONPATH=$PYTHONPATH:$pwd
export WANDB_ENTITY=openeurollm-project
export WANDB_MODE=online
python3 scripts/run_autoexp.py --config-name experiments/ville/qwen3_30B_A3B.yaml --debug $@