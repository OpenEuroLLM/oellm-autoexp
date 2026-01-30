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
Firstly, ensure you have Python loaded, activated the virtual environment and are inside `oellm-autoexp`.

1. Basic sanity check. Launch this experiment with `python scripts/run_autoexp.py --config-name experiments/diana/test_dense_50M_200MT`.

It is a very small and short experiment, meant for debugging purposes. It will test the auto-cooldown. It will work well for the stable phase and crash at the decay phase because the qos is set to debugging which doesn't allow for multiple submissions at the same time. However, the sbatch script for decay will have been created. So you can run it with sbatch.

Adjustments:
- add `legacy_tokenizer` to support old tokenizer system using the vocab and merges files
- add `wandb_entity` and `tensorbard_dir` for wandb logging 
- add `save` for checkpointing
- use `ckpt_format: torch_dist` instead of `ckpt_format: torch` because otherwise Megatron-LM will overwrite at loading time the `ckpt_step` for the last checkpoint from the `.txt` file. This happens because `ckpt_format:True` and `use_distributed_optimzier: True`. 
- since this file is for debugging autocooldown purposes, the token budget, warmup_iters, save_interval, and hyperparameter sweeps have been reduced

Notes:
- the schema default for `eval_interval` is 1000. If you also set your configurations to the same value, then the `--eval-interval` flag is silently omitted from the generated sbatch command because the command builder at megatron_backend.py:91 calls `build_cmdline_args()` with `skip_defaults=True`. This hits the logic at converter.py:244:
```
if skip_defaults and argval == spec.default:
    return []
```
Temporary fix, avoid `eval_inteval=1000`.

TO DO: set `skip_defaults=False`