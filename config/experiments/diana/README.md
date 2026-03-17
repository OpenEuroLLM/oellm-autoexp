# Steps to run the experiments

## The first time

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

## Short intro to the tool
` python scripts/run_autoexp.py --config-name experiments/diana/EXP_NAME`

To test and debug the tool, launch it with `python scripts/run_autoexp.py --config-name experiments/diana/test_dense_50M_200MT slurm.sbatch.qos=boost_qos_dbg slurm.sbatch.time="29:00"` for shorter queue times. Note that the workflow will fail when the decay run is submitted because this qos does not allow multiple submissions at the same time and the stable run is still active. After the stable run finishes, you can manually submit the already generated sbatch script for the decay run.

Since this experiment is meant for debugging purposes, the token budget, warmup_iters, save_interval, and hyperparameter sweeps have been significantly reduced.

## Run the experiments on Leonardo
Firstly, ensure you have Python loaded, activated the virtual environment and are inside `oellm-autoexp`. This branch uses this container: `/leonardo_work/OELLM_prod2026/container_images/nemo_25.11.01.sif`.

### 1. Quick experiments
Very small and quick experiments on 50M model up to 200M tokens meant for testing and debugging purposes. It takes roughly 15 minutes to train up to 200M tokens.

#### 1.1 Test autocooldown
Check that the tool works with auto-cooldown. The workflow first generates and submits the sbatch script for the stable run. A background monitoring loop then checks when the start condition for the decay run is satisfied, after which it generates and submits the sbatch script for the decay run.  Launch `python scripts/run_autoexp.py --config-name experiments/diana/test_dense_50M_200MT_autocooldown`.

#### 1.2 Test filtering + autocooldown
Launch `python scripts/run_autoexp.py --config-name experiments/diana/test_dense_50M_200MT_filtering`. The sweep contains 1 learning rate, 4 global batch sizes (16, 32, 64, 128) and 3 token budgets (50M, 100M, 200M). By default, it should produce 16 scripts (4 gbsz x (1 stable + 3 decay)), but filter out 8 of them out. The filtering conditions are 1) the warmup iters exceed 30% of the total training iters and 2) the global batch size is very small (16) AND token budget large (> 200MT) to avoid training runs with exagerately many training iterations.

#### 1.3 Test autocooldown + saving extra checkpoints before decay
Launch `python scripts/run_autoexp.py --config-name experiments/diana/test_dense_50M_200MT_ckpt_saving`. Besides the regular checkpoints, it also saves checkpoints at the iteration at which decay starts. The decay runs now start at the exact iteration before decay. For 50MT is iteration 152, for 100MT is iteration 305 and for 200MT is iteration 610.

### 2. Reproduce Niccolo's experiments
- 50M-20BT experiment: Launch `python scripts/run_autoexp.py --config-name experiments/diana/dense_50M_20BT`. This experiment runs one stable training job up to 20BT and 2 decay jobs at 12 and 20BT. The configurations are from [here](https://wandb.ai/openeurollm-project/dense_scaling/reports/Baseline-Runs-Check--VmlldzoxNTc5NDAxMQ). Results [here](https://wandb.ai/openeurollm-project/sanity-checks?nw=nwuserdianaonutu).
- 50M-50BT experiment: Launch `python scripts/run_autoexp.py --config-name experiments/diana/dense_50M_50BT`. This experiment runs one stable training job up to 50BT and 1 decay job at 50BT. Stable run is [here](https://wandb.ai/openeurollm-project/sanity-checks/runs/9egk7g5a?nw=nwuserdianaonutu) and decay run is [here](https://wandb.ai/openeurollm-project/sanity-checks/runs/v15z746d?nw=nwuserdianaonutu). Comparison between this run and Nicollo's [here](https://wandb.ai/openeurollm-project/dense_scaling/reports/Baseline-Runs-Check--VmlldzoxNTc5NDAxMQ). 
> Note, this experiment doesn't reproceduce the exact same validation loss, likely due to different validation datasets between the 2 runs (full dataset vs subsample).
- 50M-50BT_subsample experiment: Launch `python scripts/run_autoexp.py --config-name experiments/diana/dense_50M_50BT_lumi_subsample`. This experiment runs one stable training job up to 50BT and 1 decay job at 50BT. Stable run is [here](https://wandb.ai/openeurollm-project/sanity-checks/runs/3d427png?nw=nwuserdianaonutu) and decay run is [here](https://wandb.ai/openeurollm-project/sanity-checks/runs/s1lfcc0o?nw=nwuserdianaonutu). Comparison between this run and Nicollo's [here](https://wandb.ai/openeurollm-project/dense_scaling/reports/Baseline-Runs-Check--VmlldzoxNTc5NDAxMQ). 
> Note, this experiment reproceduces all metrics (train loss, validation loss) from the dense scaling run on LUMI. Important to note is that the training/validation set is a subsample of Nemotron.

#### Important adjustments
- add `legacy_tokenizer` and set it to `True` to support the old tokenizer system using the vocab and merges files
- add `wandb_entity` and `tensorboard_dir` for wandb logging 
- add `save` for checkpointing
- remove the `iter_000XX` part from `load`, since this is specified in `ckpt_step`. Otherwise it will try to find a checkpoint in `checkpoints/iter_000XXXX/iter_000XXXX`
- add padding in start condition for checkpoint folder: check if folder `iter_000XXXX` exists instead of `iter_XXXX`
- start condition only checks if the checkpoint at the correct iteration has been created and doesn't check for `latest_checkpointed_iteration.txt` since this file is by default saved in the parent directory of `checkpoints`, not inside each checkpoint folder
- use `ckpt_format: torch_dist` instead of `ckpt_format: torch` because otherwise Megatron-LM will overwrite at loading time the `ckpt_step` with the last checkpoint from the `latest_checkpointed_iteration.txt` file. This happens because `ckpt_format: torch` and `use_distributed_optimzier: True`. 

### 3. Reproduce multilingual experiment
Reproduce multilingual experiments: 600M model on 100B tokens using data mix option 4 (near natural distribution). Launch `python scripts/run_autoexp.py --config-name experiments/diana/dense_600M_100BT_multilingual_option2 "backend.megatron.data_path=[$(cat /leonardo_work/OELL
M_prod2026/users/fvitiugi/training/datamix-option2.txt | tr ' \n' ',,')]" `. Results [here](https://wandb.ai/openeurollm-project/sanity-checks/runs/jaquzvof) and original run [here](https://wandb.ai/openeurollm-project/1TTest/runs/mghf79yo?nw=nwuserdianaonutu). There is a small discrepancy. Namely, my run had used biases and it shouldn't have: `disable_bias_linear: True`, however, it should have been `disable_bias_linear: False`. Comparison between this run and Fedor's [here](https://wandb.ai/openeurollm-project/dense_scaling/reports/Baseline-Runs-Check--VmlldzoxNTc5NDAxMQ).

### 4. Dense scaling setup

For reproducing Niccolo's English dense scaling runs launch `python scripts/run_autoexp.py --config-name experiments/diana/dense_50M`. It contains the full sweep, including filtering conditions for runs in which the warmup iters exceed 30% of the total training iters and for runs in which the global batch size is very small (16, 32) and token budget large (> 50BT).

### 5. Qwen-like architecture

`python scripts/run_autoexp.py --config-name experiments/diana/dense_qwen_150M`. To be adjusted: data path, tokenizer path, optimal hparams, checkpoint saving. 

## Open TO DOs
- change Megatron to save checkpoints right before the LR decay phase
- adjust the experiment config to use the checkpoints before LR decay phase