# Steps to run the experiments

## The first time:

1. Load Python `module load python/3.11.7`
2. Create a virtual environment:
```
cd $HOME
python -m venv my_venv
source my_venv/bin/activate
```
3. Clone the repository:
 ```
 cd $WORK/users/donutu00
 git clone https://github.com/OpenEuroLLM/oellm-autoexp.git --recurse-submodules
 cd oellm-autoexp
 ```
4. Install it and switch to my branch:
``` 
pip install -e .
bash ./apply_megatron_numpy_product_patch.sh
git checkout exp_diana
```
5. Make sure you have the relevant [environment variables](https://github.com/OpenEuroLLM/oellm-autoexp?tab=readme-ov-file#environment-variables) set in your `$HOME/.bashrc`. 


## Run the experiments
Firstly, ensure you have Python loaded, activated the virtual environment and are inside `oellm-autoexp`. This branch uses this container: `/leonardo_work/OELLM_prod2026/container_images/nemo_25.11.01.sif`.

1. Quickly check that the tool works with auto-cooldown. Launch this experiment with `python scripts/run_autoexp.py --config-name experiments/diana/test_dense_50M_200MT`.

It is a very small and short experiment, meant for debugging purposes (max 30 min per sbatch script). It tests the auto-cooldown. It will work well for the stable phase and crash at the decay phase because the qos is set to debugging which doesn't allow for multiple submissions at the same time. However, the sbatch script for decay will have been created. So, you can run it with sbatch. This script has also been tested without the debugging qos, and auto-cooldown works as expected.

2. Reproduce Niccolo's experiments launching `python scripts/run_autoexp.py --config-name experiments/diana/dense_50M_20BT`. This experiment is meant to run one stable run up to 20BT and decay at 12 and 20 BT. The configurations are from [here](https://wandb.ai/openeurollm-project/dense_scaling/reports/Baseline-Runs-Check--VmlldzoxNTc5NDAxMQ). Note: this experiment is currently running.

Adjustments from the `main` branch:
- add `legacy_tokenizer` and set it to `True` to support the old tokenizer system using the vocab and merges files
- add `wandb_entity` and `tensorbard_dir` for wandb logging 
- add `save` for checkpointing
- remove the `iter_000XX` part from `load`, since this is specified in `ckpt_step` and otherwise it will try to find a checkpoint in `checkpoints/iter_000XXXX/iter_000XXXX`
- add padding in start condition for checkpoint folder: check if folder `iter_000XXXX` exists instead of `iter_XXXX`
- start condition only checks if the checkpoint at the correct iteration has been created and doesn't check for `latest_checkpointed_iteration.txt` since this file is by default saved in the parent directory of `checkpoints`, not inside each checkpoint folder
- use `ckpt_format: torch_dist` instead of `ckpt_format: torch` because otherwise Megatron-LM will overwrite at loading time the `ckpt_step` with the last checkpoint from the `latest_checkpointed_iteration.txt` file. This happens because `ckpt_format:True` and `use_distributed_optimzier: True`. 
- since this file is for debugging auto-cooldown purposes, the token budget, warmup_iters, save_interval, and hyperparameter sweeps have been significantly reduced

Notes:
- the schema default for `eval_interval` is 1000. If you also set your configurations to the same value, then the `--eval-interval` flag is silently omitted from the generated sbatch command because the command builder at `megatron_backend.py` calls `build_cmdline_args()` with `skip_defaults=True`. This hits the logic at `converter.py`:
```
if skip_defaults and argval == spec.default:
    return []
```
Temporary fix, avoid `eval_inteval=1000`. Perhapds to do: set `skip_defaults=False`
- TO DO: implement saving and loading the precise checkpoints at which decay should start
- TO DO: test filtering of runs