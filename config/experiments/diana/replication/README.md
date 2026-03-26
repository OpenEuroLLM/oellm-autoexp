## Configurations in this folder

The purpose of this folder is to validate the `oellm-autoexp` tool by using it to replicate Niccolo's experiments that were originally run with the `autoexp` tool.

All configs use:
- **Backend**: Megatron + torchrun on Leonardo
- **WandB project**: `sanity-checks` under `openeurollm-project`

---

## English Dense Scaling Experiments

### `dense_50M.yaml` — Replication of Niccolo's full hyperparameter sweep (up to 300BT)

Full composable sweep over 5 LR values × 7 batch sizes × (1 stable phase up to 300B tokens + 9 decay stages at 6B/12B/20B/30B/50B/80B/120B/200B/300BT), yielding ~337 configs after filtering. Decay stages use 20% cooldown with WSD scheduler and load from the stable phase checkpoint. Data: Nemotron-CC 1.0 high-quality.

Filters exclude runs where warmup exceeds 30% of total training iterations, and runs with very small batch sizes (gbsz 16 or 32) at large token budgets (> 50BT).

```bash
python scripts/run_autoexp.py --config-name experiments/diana/replication/dense_50M
```

### `dense_50M_20BT.yaml` — Replication of Niccolo's 20BT experiment

Single stable run up to 20BT + 2 decay jobs at 12BT and 20BT. Fixed LR=1e-3, gbsz=256, micro_batch_size=16. Data: Nemotron-CC 1.0 high-quality. Reference configurations from [dense_scaling baseline report](https://wandb.ai/openeurollm-project/dense_scaling/reports/Baseline-Runs-Check--VmlldzoxNTc5NDAxMQ). Results [here](https://wandb.ai/openeurollm-project/sanity-checks?nw=nwuserdianaonutu).

```bash
python scripts/run_autoexp.py --config-name experiments/diana/replication/dense_50M_20BT
```

### `dense_50M_50BT.yaml` — Replication of Niccolo's 50BT experiment (full dataset)

Single stable run up to 50BT + 1 decay job at 50BT. Fixed LR=1e-3, gbsz=256, micro_batch_size=16. Data: Nemotron-CC 1.0 high-quality. Stable run [here](https://wandb.ai/openeurollm-project/sanity-checks/runs/9egk7g5a?nw=nwuserdianaonutu), decay run [here](https://wandb.ai/openeurollm-project/sanity-checks/runs/v15z746d?nw=nwuserdianaonutu). Comparison with Niccolo's run [here](https://wandb.ai/openeurollm-project/dense_scaling/reports/Baseline-Runs-Check--VmlldzoxNTc5NDAxMQ).

> Note: does not reproduce the exact same validation loss, likely due to a different validation dataset (full dataset vs subsample).

```bash
python scripts/run_autoexp.py --config-name experiments/diana/replication/dense_50M_50BT
```

### `dense_50M_50BT_lumi_subsample.yaml` — Replication of Niccolo's 50BT experiment (LUMI subsample)

Same setup as `dense_50M_50BT.yaml` but uses the copied subsample dataset from LUMI (`Nemotron-cc-2024-HQ-LUMI-sample-318BT/high-all`) instead of the full dataset — matching the dataset used in the original LUMI run. Stable run [here](https://wandb.ai/openeurollm-project/sanity-checks/runs/3d427png?nw=nwuserdianaonutu), decay run [here](https://wandb.ai/openeurollm-project/sanity-checks/runs/s1lfcc0o?nw=nwuserdianaonutu). Comparison with Niccolo's run [here](https://wandb.ai/openeurollm-project/dense_scaling/reports/Baseline-Runs-Check--VmlldzoxNTc5NDAxMQ).

> Note: successfully reproduces train and validation loss from the LUMI dense scaling run.

```bash
python scripts/run_autoexp.py --config-name experiments/diana/replication/dense_50M_50BT_lumi_subsample
```

---

## Multilingual Dense Scaling Experiments

### `dense_600M_100BT_multilingual_option2` — Replication of Fedor's 600M multilingual experiment

600M model on 100B tokens using data mix option 2 (near-uniform distribution with 50% of translated data). Results [here](https://wandb.ai/openeurollm-project/sanity-checks/runs/jaquzvof), original run [here](https://wandb.ai/openeurollm-project/1TTest/runs/mghf79yo?nw=nwuserdianaonutu). Comparison with Fedor's run [here](https://wandb.ai/openeurollm-project/dense_scaling/reports/Baseline-Runs-Check--VmlldzoxNTc5NDAxMQ).

> Note: small discrepancy — `disable_bias_linear` was incorrectly set to `True` (biases were used); it should have been `False`.

```bash
python scripts/run_autoexp.py --config-name experiments/diana/dense_600M_100BT_multilingual_option2 \
  "backend.megatron.data_path=[$(cat /leonardo_work/OELLM_prod2026/users/fvitiugi/training/datamix-option2.txt | tr ' \n' ',,')]"
```
