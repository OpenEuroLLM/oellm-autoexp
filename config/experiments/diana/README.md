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

1. Simple check:`test1_dense_50M_200MT,yaml`. TO DO: check why the naming of the job output directory doesn't work with stages
TO DO: Update with refactored configs.

Adjustments:
- add `legacy_tokenizer` to support old tokenizer system using the vocab and merges files
- add `wandb-entity` and `tensorbard_dir` for wandb logging 
- add `save` for checkpointing
- since this file is for debugging autocooldown purposes, the token budget, warmup_iters, save_interval, and hyperparameter sweeps have been reduced