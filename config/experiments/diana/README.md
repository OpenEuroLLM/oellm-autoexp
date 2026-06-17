# Steps to run the experiments


## Running experiments on Leonardo

Before running, ensure you have Python loaded, the virtual environment activated, and are inside `oellm-autoexp`. This branch uses the container:
`/leonardo_work/OELLM_prod2026/container_images/nemo_25.11.01.sif`

```
module load python/3.11.7 && source $HOME/my_venv/bin/activate
```

### Basic usage

Running the tool performs three phases in sequence: **plan** → **submit** → **monitor**. Since monitoring must stay alive after you disconnect, run this inside a **tmux session**:

```
tmux new-session -s autoexp
module load python/3.11.7 && source $HOME/my_venv/bin/activate
python scripts/run_autoexp.py --config-name experiments/diana/EXP_NAME
```

To reattach later:
```
tmux attach-session -t autoexp
```

> **Important:** Leonardo has multiple login nodes. You must always log in to the **same login node** where you created the tmux session, otherwise the session will not be found.

### Monitoring

The monitoring loop runs as part of `run_autoexp.py`, but it can also be restarted independently at any time — for example, if the tmux session was closed due to maintenance. Restarting monitoring will resume where it left off, including triggering any decay runs that were pending when the session died.

Run it in its own tmux session (same login-node caveat applies):
```
tmux new-session -s monitor
module load python/3.11.7 && source $HOME/my_venv/bin/activate
python scripts/monitor_autoexp.py --session-dir ./monitor_state/<session_id>
```

### Syncing W&B runs

Syncing all offline runs for the given experiment in one go. Use the provided sync script, passing the experiment folder name:
```
python scripts/sync_runs.py --folder qwen3_300M_gpt_neox
```


### Calculating GPU hours

To compute the total GPU hours consumed by an experiment, pass the experiment folder:
```
python scripts/gpu_hours.py results/qwen3_600M_gpt_neox/
```

### Testing and debugging

Use the debug QoS for shorter queue times:
```
python scripts/run_autoexp.py --config-name experiments/diana/testing/test_dense_50M_200MT_autocooldown \
    slurm.sbatch.qos=boost_qos_dbg slurm.sbatch.time="29:00"
```

> **Note:** The workflow will fail when the decay run is submitted because `boost_qos_dbg` does not allow multiple simultaneous submissions. After the stable run finishes, manually submit the already-generated sbatch script for the decay run.


---
## First-time setup

1. Load Python:
   ```
   module load python/3.11.7
   ```

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

4. Switch to the experiment branch, install, and apply patches:
   ```
   git checkout exp_diana
   pip install -e .
   bash ./apply_megatron_numpy_product_patch.sh
   ```

5. Make sure you have the relevant [environment variables](https://github.com/OpenEuroLLM/oellm-autoexp?tab=readme-ov-file#environment-variables) set in your `$HOME/.bashrc`.

---

## Development

### Integrating submodule changes

After pulling changes that update a submodule, run from the root of `oellm-autoexp`:
```
git submodule update --init --recursive
```

### Merging to main

Before merging a branch into main, install:
```
pre-commit install
```

---

## Config developements

- `legacy_tokenizer: True` — supports the old tokenizer system (vocab and merges files)
- Added `wandb_entity` and `tensorboard_dir` for logging
- Added `save` for checkpointing
- Removed the `iter_000XX` suffix from `load` — it is specified separately via `ckpt_step`; keeping it would cause Megatron to look for `checkpoints/iter_000XXXX/iter_000XXXX`
- Start condition checks for zero-padded folder name `iter_000XXXX` instead of `iter_XXXX`
- Start condition checks only for the checkpoint folder, not for `latest_checkpointed_iteration.txt` (which is saved in the parent of `checkpoints/`, not inside each checkpoint folder)
- `ckpt_format: torch_dist` instead of `ckpt_format: torch` — prevents Megatron from overwriting `ckpt_step` with the value from `latest_checkpointed_iteration.txt` (this interaction only occurs when `use_distributed_optimizer: True`)

---

## TODO

- Integrate within oellm-autoexp the common errors encountered during experiments and add restart criteria
- Create a CSV-based run state overview (as shown by Niccolo) to more easily monitor the status of all runs
