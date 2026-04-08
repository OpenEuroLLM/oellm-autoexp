# Testing configurations

Small, fast experiments on a 50M dense model trained up to 200M tokens, meant for testing and debugging the `oellm-autoexp` tool. Each run takes roughly 15 minutes. Model architecture and other hyperparameter configs follow Niccolo's dense monolingual setup.

All configs use:
- **Backend**: Megatron + torchrun on Leonardo
- **Data**: Nemotron-CC 1.0 high-quality split (real + several synthetic subsets), GPT-NeoX-20B tokenizer
- **Scheduler**: WSD with linear decay, `lr_warmup_iters=200` (except 1.3 which uses 30)
- **WandB project**: `sanity-checks` under `openeurollm-project`

---

#### 1.1 Test autocooldown — `test_dense_50M_200MT_autocooldown`

Validates the pull-based multi-stage workflow end-to-end with a minimal sweep (1 LR, 1 batch size, 1 token budget).

The tool submits the stable run first, then a background monitoring loop polls until the required checkpoint exists (`FileExistsCondition`) and automatically submits the decay run. The decay run also registers a `LogPatternCondition` that cancels it if the stable job crashes.

- Stable phase: 200M tokens, no decay (`lr_wsd_decay_iters=0`)
- Decay phase: branches off at 80% of stable training (20% decay fraction), loads checkpoint from the stable run at the nearest `save_interval`-aligned iteration

```bash
python scripts/run_autoexp.py --config-name experiments/diana/testing/test_dense_50M_200MT_autocooldown
```

---

#### 1.2 Test sweep filtering + autocooldown — `test_dense_50M_200MT_filtering`

Validates sweep filtering logic on top of the autocooldown workflow. Tests that ill-conditioned hyperparameter combinations are correctly pruned before job submission.

- Sweep: 1 LR × 4 global batch sizes (16, 32, 64, 128) × (1 stable + 3 decay token budgets: 50M, 100M, 200M) = **16 scripts** before filtering
- Filter condition 1: warmup iters > 30% of total training iters → removes runs where warmup dominates training
- Filter condition 2: `global_batch_size=16` AND `tokens > 100M` → removes runs with excessively many iterations for small batch sizes
- Expected outcome: **8 scripts filtered out**, **8 scripts submitted**
- The stable stage is always kept when its own filter check passes, since decay stages depend on it

```bash
python scripts/run_autoexp.py --config-name experiments/diana/testing/test_dense_50M_200MT_filtering
```

---

#### 1.3 Test autocooldown + extra checkpoint saving — `test_dense_50M_200MT_ckpt_saving`

Validates that the stable run saves additional checkpoints at the exact iterations where each decay branch will start, so decay runs can resume from the precise (non-rounded) branching point.

Differences from 1.1:
- `lr_warmup_iters=30` (reduced to allow for extra checkpointing)
- `save_interval=50` (more frequent saves)
- `save_extra_steps` saves checkpoints at the 80% mark for each of the three token budgets: iteration 152 (50MT), 305 (100MT), 610 (200MT)
- Decay runs load from the exact `start_iter` (not the rounded `start_iter_round` used in 1.1/1.2)

```bash
python scripts/run_autoexp.py --config-name experiments/diana/testing/test_dense_50M_200MT_ckpt_saving
```