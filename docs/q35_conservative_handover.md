# Conservative (Qwen3.5-35B-A3B-like) run: handover (2026-10-27)

The "conservative" baseline next to the "ambitious" hybrid (config/experiments/korbi/production/
hybrid_main_mn5.yaml). Set up and validated in session oellm-autoexp-hybrid-c2; to be run by the
main campaign agent from here.

## State

- **Nothing of it runs.** `run_2` (MHA by mistake, see below) ended at step 10,100 (loss 1.53) when
  the user cancelled it on 2026-10-26; its checkpoints are kept as an MHA reference
  (`/gpfs/scratch/ehpc533/hybrid_nucleus/q35_conservative/run_2/main/ckpt/iter_0001000 .. 0010000`).
- **Next:** start `main_gqa` fresh (GQA, step 0) on 128 nodes. Not submitted yet: the user decides
  the account (ehpc533 or cns144, both `acc_xehpc`).
- Pending leftover: job 48094115 (`q35c_run_3_128n`, cns144) is a **queue-time probe only** (old
  suite name). `monitor_logs/q35_run3_queue_probe.sh` logs its start and cancels it. Cancel it when
  `main_gqa` is submitted.

## Config

- `config/experiments/korbi/production/q35_conservative_mn5.yaml` (base),
  `q35_conservative_32_mn5.yaml` / `q35_conservative_128_mn5.yaml` (cns144) /
  `q35_conservative_128e_mn5.yaml` (ehpc533). Suite `main_gqa`; job names `q35c_main_gqa_{32n,128n,128e}`;
  output `hybrid_nucleus/q35_conservative/main_gqa/main`; per-job-name sbatch / current-log paths
  (several jobs share the output directory). Megatron >= bdbda626a.
- Model: 40 blocks, pattern `*ELELE*ELELELE...*E` (11 gated-attention + 29 GDN, block 1 attention),
  every block MoE: 256 experts x 512, top-8, sigmoid + expert bias (renormalised top-8), gated
  shared expert 512; hidden 2048; attention 16 Q heads x 256, **GQA 2 KV groups**, partial RoPE 0.25,
  base 1e7, qk-norm, zero-centred RMSNorm; untied 262k embedding (OELLM tokenizer). 34.71B total,
  ~2.4B active (non-embedding).
- Training: GBS 2048 x 8k (16.8M tokens; **user: never above 16M tokens per step**), flag-decon-v1
  data with document masking + BOS (index reused from run_2), AdamW 1e-3, 2k warm-up, constant LR,
  WD 0.1, clip 1.0, bf16. Expert metrics (per-layer logging + viability) on, log_interval 1 (free).
- Parallelism: Megatron-FSDP + MoonEP EP4, mbs 4, **hybrid sharding: ZeRO-3 inside 8-node groups**
  (`num_distributed_optimizer_instances = nodes / 8`).
- Checkpoints: rolling every 100 steps (>= 2 kept), persistent every 1000 (~390 GB each; **user:
  keep all for now**; scratch ~6 TB/day at 128 nodes; revisit by ~2026-11-01). Async saves +
  persistent worker + `ckpt_assume_constant_structure` (amb 32n: ~95 s blocking for the first two
  saves after a (re)start, then ~7 s). Peer loop hard-links eval marks (steps 2000 * 2^n) into
  `ckpt_eval/`: never delete. Exit signal `SIGUSR2` (by name).

## Speed (tok/s/GPU, expert metrics on)

| layout | 8n | 16n | 32n | 128n |
|---|---:|---:|---:|---:|
| FSDP ZeRO-3 over all ranks | 6,457 | 6,265 | 4,385 | - |
| **FSDP HSDP, 8-node groups** | | ~6,530 | **6,449-6,489** | **~4,200** |
| PP4 x VPP2 (colleague's) | 6,077 | | 5,902 | 4,204 |

At 128 nodes every layout tested lands at ~4.2k (HSDP 8- or 16-node groups, outer-sharded optimizer,
16 NCCL channels, PP): a per-step floor at 1 micro-batch per GPU; only a larger GBS helped (ruled out).
Wall clock is still ~2.5x the 32-node rate. Logs: `q35_conservative/{f8,d8,s32,s128x}`.

## Verified (all on MN5)

- Checkpoint save/resume incl. layout changes: 8n -> 16n (FSDP, PP), FSDP -> HSDP, rolling
  (non-persistent) fsdp_dtensor checkpoints (Megatron 00dc9402f), SIGUSR2 save + exit
  (`scripts/korbi/production/signal_save_exit.sh`: signal torchrun's children only; `scancel
  --signal=USR2 --full` kills the job without a save), async saves + resume (16n).
- Model audit of the GQA config from checkpoint metadata: every tensor shape as intended (see
  memory / session notes); MHA-GQA difference exactly 161,480,704 parameters.

## Pitfalls hit (and fixed)

- GQA silently off: Megatron applies `num_query_groups` only with `group_query_attention`.
- MoonEP checkpoint replica ids (44a703806), fsdp_dtensor rerun/RNG state on layout change
  (ea2df1aaf), rolling fsdp_dtensor loads (00dc9402f), Adam step counters on resume (780202c8a).
- Shared job.sbatch across jobs in one output dir (a restart could submit the other scale).
- Bad nodes (excluded in `mn5_exclude_nodes.txt`): as01r4b20 (SIGBUS at load), as01r1b03 and
  as07r3b14 (set_device failures).
- A colleague (W&B user tvosch, account surf593435) ran a separate copy of the old `run_1` config
  (MHA, old data) to step 3,712; W&B `flag_hybrid` "run_1_q35_main" is that, not ours. Ours:
  W&B id `conservative_q35_run2` (run_2); main_gqa needs a line in
  `scripts/korbi/wandb_sync/runs.tsv`.

## Start main_gqa (128 nodes, ehpc533)

```
./sync_to_marenostrum.sh q35gqa
ssh marenostrum_new 'cd $R/autoexp_q35gqa && scripts/korbi/mn5_autoexp_bg.sh q35c_main_gqa_128e \
    scripts/run_autoexp.py --config-name experiments/korbi/production/q35_conservative_128e_mn5 --yes'
```
Dry-run first and parse the rendered arguments in the container (parse_args on the args between
`pretrain_hybrid.py` and the closing quote): catches argument errors before a 128-node start.
