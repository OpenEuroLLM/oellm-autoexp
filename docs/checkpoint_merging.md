# Weighted checkpoint merging (WSM)

Averaging a window of consecutive checkpoints gives a stronger, lower-variance
quality signal than any single checkpoint, without paying for a learning-rate
decay run at every evaluation point. NVIDIA used this to select Nemotron 3 Super
(+2-4 points across a 12-benchmark suite; the shipped base model was itself a
500B-token merge), and the WSM paper reports +3.5 on MATH, +2.9 on HumanEval and
+5.5 on MMLU-Pro over WSD.

The merge is done by `tools/checkpoint/weighted_merge.py` from Megatron-LM
(PR #5114, merged 2026-09-16). `scripts/merge_checkpoints.py` in this repository plans
the windows and submits the Slurm jobs.

## Merge before converting to HF

The merge reads and writes `torch_dist` and rejects every other format, so it
sits upstream of HF export:

```
iter_* (torch_dist) -> weighted_merge -> merged iter_* (torch_dist) -> convert + validate -> HF branch
```

Doing it on this side also matters for correctness and cost:

- Transformer Engine `_extra_state` (FP8 amax history and scales) is **copied**
  from one chosen source rather than averaged, because averaging scale history
  is meaningless. That state has no representation in safetensors, so merging
  after conversion silently discards a decision the tool makes explicitly.
- A merged `torch_dist` checkpoint stays loadable for evaluation or resume
  (`--no-load-optim --no-load-rng`). An averaged safetensors file can only be
  evaluated.
- Merging is CPU-only and I/O-bound; conversion needs a GPU per checkpoint.
  Merging first and converting only the merged points is far cheaper than
  converting everything and averaging afterwards.

## Choosing the window

The WSM paper identifies **merge duration -- the training window -- as the most
critical factor, ahead of both checkpoint interval and merge quantity.** Pick the
duration first; how many checkpoints sit inside it matters much less.

At `global_batch_size 4096 x seq_length 4096` the 32B runs see 16.78M tokens per
iteration, and their checkpoints sit on a 2000-iteration grid:

| Window | Iterations | Tokens |
|---|---|---|
| 8 checkpoints | 16,000 | 268B |
| 15 checkpoints | 30,000 | 503B (matches the Nemotron final merge) |
| 30 checkpoints | 60,000 | 1.01T |

The number of merged points in a series is a separate knob, the `--stride`
between window endpoints. Over b1's 93 on-grid checkpoints with a 15-wide window,
stride 1 gives 79 merges, stride 5 gives 16, stride 10 gives 8. The series is
anchored at the newest checkpoint and walks back.

## Obtain the merge tool

Pinned to the commit the PR merged as:

```bash
curl -fsSLO https://raw.githubusercontent.com/NVIDIA/Megatron-LM/636a02299d20b95afa696af14c625a1398714578/tools/checkpoint/weighted_merge.py
echo "1d9230ce0aada5f3abc78bc8f6a9241a3f3437770f972a218d7c0424346589a7  weighted_merge.py" | sha256sum -c
```

The tool imports only `megatron.core.dist_checkpointing`,
`megatron.core._rank_utils.safe_get_rank` and
`megatron.core.dist_checkpointing.core.maybe_load_config` -- all present in the
B1 pin `799ff4e6e1c3` -- and no model code, so it runs against our own pinned
Megatron without adopting upstream. Point `--megatron-root` at that tree, or omit
it to use whatever the container image ships.

## Pilot: prelude on Leonardo

Prelude is the cheapest way to validate the whole chain, and it already contains
the comparison the technique claims: 421 stable-phase checkpoints on a 2400
grid, then a real `anneal300b` decay phase (955200 -> 989075). Merging the stable
phase ending at 955200 and scoring it against the annealed model is WSM vs WSD
measured on our own model rather than taken on faith.

```bash
uv run python scripts/merge_checkpoints.py \
    --checkpoints-dir /leonardo_scratch/fast/OELLM_prod2026/production_training/baby_9b_dense/checkpoints \
    --output-dir /leonardo_scratch/large/userexternal/$USER/prelude-merged \
    --run-label prelude --min-iteration-interval 2400 \
    --window 16 --end-iteration 955200 \
    --merge-tool $PWD/weighted_merge.py \
    --container-image /leonardo_work/OELLM_prod2026/container_images/<image>.sif \
    --singularity-bind /leonardo_scratch --singularity-bind /leonardo_work \
    --account OELLM_prod2026 --partition boost_usr_prod --qos boost_qos_dbg \
    --gres gpu:1 --nodes 1 --ntasks-per-node 8 --time-limit 01:00:00 \
    --dry-run
```

Run with `--dry-run` first. It prints the planned windows and passes `--dry-run`
to the merge tool, which validates every source's layout and prints the resolved
input list without writing anything.

Then convert the merged series as an ordinary run, using the **source** run's
training config -- a merged checkpoint has no `logs/` of its own:

```bash
uv run python scripts/mass_convert_checkpoints.py \
    --checkpoints-dir /leonardo_scratch/large/userexternal/$USER/prelude-merged/prelude_wsm16_minus-sqrt/checkpoints \
    --training-config /leonardo_scratch/fast/OELLM_prod2026/production_training/baby_9b_dense/logs/current.yaml \
    --run-label preludewsm16 --output-dir <staging> ...
```

Pass the remaining flags -- container image, binds, account, Slurm placement and
the Hub target -- exactly as for an unmerged run; see
`scripts/mass_convert_checkpoints.py --help`. A distinct `--run-label` is what
keeps merged branches separable from trained ones on the Hub.

## Applying it to the 32B

Same shape against a 2000 grid, planning several durations at once so they can be
compared:

```bash
uv run python scripts/merge_checkpoints.py \
    --checkpoints-dir <run>/checkpoints \
    --output-dir /e/scratch/e-sta-openeurollm/$USER/oellm-32b-merged \
    --run-label b1 --min-iteration-interval 2000 \
    --window 8 --window 15 --window 30 --stride 5 \
    --merge-tool $PWD/weighted_merge.py \
    --container-image /e/project1/e-sta-openeurollm/container/<image>.sif \
    --singularity-bind /e \
    --account e-sta-openeurollm --partition booster --qos normal \
    --gres gpu:4 --nodes 1 --ntasks-per-node 8 --time-limit 02:00:00
```

Every cluster we have access to offers only GPU partitions, so the allocation
requests a GPU that the merge never uses. Size `--ntasks-per-node` to saturate
storage read bandwidth rather than to match GPUs.

## Gotchas

- **Optimizer state is rejected by default.** Our checkpoints carry distributed
  optimizer state, and the tool refuses sharded entries outside the model roots
  rather than ignoring them. `scripts/merge_checkpoints.py` always passes
  `--merge-ignore-non-model-state`, which is the explicit opt-out.
- **Identical layout required** -- same tensor keys, shapes, chunk/sharding and
  dtype. True within a run, never across runs. b1 changed Megatron source
  mid-run at 192k, so dry-run any window spanning that boundary before trusting
  it.
- **Off-grid checkpoints are excluded as endpoints.** Segment-boundary saves such
  as `iter_0084238` sit a few hundred iterations from their neighbour and would
  over-weight that point. `--min-iteration-interval` filters them out.
- **`_extra_state` source defaults to the earliest input.** For a checkpoint
  published as `iter_<END>` the endpoint is usually the better source; a dry-run
  prints the resolved input count, so set `--extra-state-source-index`
  explicitly once that count is known.
- **No verify-load step.** Merge correctness verification is explicitly out of
  scope for the tool. The conversion pass that follows (test load, generation,
  vocabulary check) is what covers it.
- **Merged checkpoints are model-only**, so they are far smaller than the source
  training checkpoints and load with `--no-load-optim --no-load-rng`.
