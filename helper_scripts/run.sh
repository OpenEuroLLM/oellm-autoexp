set -eo pipefail
module purge
module load Stages/2026
module load GCCcore/14.3.0
module load Python/3.13.5
source /e/project1/e-sta-openeurollm/komulainen1_jupiter/oellm-autoexp/.venv/bin/activate
export WANDB_ENTITY=openeurollm-project
python3 scripts/run_autoexp.py --submit-and-exit --config-name experiments/ville/jupiter/jupiter_qwen-30B.yaml $@