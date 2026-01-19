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
Experiments:
1. Generate sbatch scripts based on hyperparameter sweep and multi-stage. Tested only on generating scripts (doesn't run them):
```
PYTHONPATH=. python scripts/run_autoexp.py --config-ref experiments/diana/korbi_dense_50M_50BT_pull_leonardo --dry-run
```
2. Run the stable-decay stages. Simplified the hyperparameters configs. Generate and run the scripts (under testing): 
```
PYTHONPATH=. python scripts/run_autoexp.py --config-ref experiments/diana/korbi_dense_50M_50BT_pull_leonardo_simple_stable_decay
```
3. Related to point (1, generate scripts), but with additional filtering of experiments. To be tested with the latest fixes.

# The experiments  
Use as a baseline `korbi/korbi_dense_50M_50BT_pull` and make the following adjustments:
- in `config/slurm/leonardo.yaml` added `WANDB_MODE: "offline"`
- adjusted defaults wrt machine: from lumi -> leonardo
- added leonardo container settings. Container image used: `nemo_25.11.01.sif`
- adjusted `data_path` to leonardo data path (this is the old, problematic data. To be changed for actual runs)
- removed `tokenizer_type` and `tokenizer_model` and added `vocab_file` and `merge_file` paths
- added `data_cache_path` and `wandb_save_dir`
- adjusted model from 300M -> 50M
- adjusted hyperparameters with the latest design choices from xmas experiments: `adam_beta2: 0.99`, `min_lr: 1e-5`, `lr_warmup_iters: 2000`, `eval_interval: 3200`, `save_interval: 8000`.

