# Checkpoint merging

`merge_hf.py` averages HF safetensors exports of one architecture (the bf16 exports that
`scripts/downstream_eval/eval_checkpoints.sh convert` writes) into a new export. Needs only numpy;
streams per tensor chunk (256 MiB), so memory stays flat for any model size.

```bash
python3 merge_hf.py --root <export root> --out v2anneal_wma5_112k_120k --weights linear \
    v2anneal_112k v2anneal_114k v2anneal_116k v2anneal_118k v2anneal_120k   # oldest first
```

- `--weights uniform | linear | a,b,...` (linear = 1:2:..:n, newest heaviest).
- The result is the correctly rounded weighted mean: float64 sum with the raw weights, one
  division, one round-to-nearest-even. Independent of input order.
- Writes `<out>.partial`, then renames; `MERGE_INFO.json` (inputs, weights, `update_scale`) and
  `.convert_done` mark a finished merge, so the FLAG `prepare` stage can wait for it. Re-running a
  finished merge is a no-op; an interrupted one is redone.
- 32B: ~4 min for 3 inputs on a JUPITER compute node, I/O bound (n x 64 GB read, 64 GB written).

`update_scale[i]` is the factor the merge effectively applies to the training updates between
input i-1 and i: merging with weights c_i equals keeping the first input and scaling those
updates by sum_{k>=i} c_k (WSM, arXiv:2507.17634). So a merge inside an LR cooldown is extra LR
decay after the fact; at constant LR it stands in for a cooldown.

## As an oellm-autoexp chain (merge -> FLAG suite)

`config/experiments/oellm_32b_dense/anneal_merge5_evals.yaml` is the worked example. It reuses
`lineage_flag_evals`' convert -> prepare -> eval array -> collect chain and swaps the convert
command by way of `aux.export_command`. For new merges, add entries `{name, inputs, weights, exists:
false, ...}` to `aux.lineage_ckpts` and select them with `aux.ckpt_select`.

Tests: `tests/unit/test_merge_hf.py` (needs torch + safetensors; runs in CI's megatron-extras job).
Real-data check against existing exports: `dump/model_merging/verify_against_merge3.py`.
