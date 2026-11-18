# v4b: the iteration-36000 crash and everything it took to run the arm

Status: **fixed and verified.** Run `1579` on host008 resumed at `non-persistent iter 35999`, passed
iteration 36000 with 0 `misaligned`, and printed arch metrics, validation losses and a checkpoint
save. This note lists every fix, in the order they were needed, with the evidence for each.

## 1. Symptom

Every v4b attempt died at iteration 36000 with `CUDA error: misaligned address`, on about 1900
ranks, `rc 137`, and no `arch metrics at iteration` line. Iterations 35997–35999 were clean.
36000 is both the first `arch_metrics_interval: 500` step and an `eval_interval: 1000` step.

## 2. What isolated the cause

| observation | conclusion |
| --- | --- |
| v4a has no V layers at all | v4a only proves the generic arch path works |
| v4c reaches 36000, prints `arch metrics at iteration 36000`, has **0 misaligned** | the fault is not the V layers, mHC, validation or the arch-metrics module as such |
| v4c differs only by `v_block_gate_ramp: 100` / `v_block_learnable_gate: false` | the fault is in the learnable-gate path |
| the fault survives every DTensor-safe change to the metric recording | the recording itself is not where it dies |
| `CUDA_LAUNCH_BLOCKING=1` gives frames (run 1577) | the fault is in the **unfused mHC coefficients** |

The blocking run produced the chain:

```
transformer_layer.py:1126  forward
transformer_layer.py:1226  _mhc_read   ->  h_pre, h_post, h_res = mixer(streams)
recompute.py:166           custom_forward
random.py:671 / 550        backward / _fork_rng
```

## 3. Root cause

`_record_v_gate` needs `h_pre` for `read_share`. The fused read does not return it:
`mhc_fused.coeffs_and_read` returns `h_pre = None` ("the fused read keeps it"). So on arch-metrics
steps `_mhc_read` deliberately takes the **unfused** `MHCMixer.forward`, which calls
`_mhc_coeffs_compiled` — a `torch.compile`d (Inductor) kernel.

Inductor-generated code assumes aligned pointers. Under activation recompute the buffers are freed
and reallocated, so the **recomputed** forward hands that kernel an unaligned address. Hence: only on
metrics steps, and surfacing in the recompute backward rather than at the call site.

## 4. Code fixes

| commit | change | why |
| --- | --- | --- |
| `1f7bcd072` (super `4cb18878`) | `MHCMixer.forward(streams, eager=False)`; `_mhc_read` passes `eager=arch_metrics.active()` | **the actual fix**: the metrics path runs the same mHC math without the compiled kernel; the training path keeps it |
| `1a254418e` (super `429a61ba`) | `_plain_gate` / `_plain_weight`; stop-once guard with one warning | `arch_metrics` keeps plain CUDA buffers and all-reduces them, so a DTensor must never be recorded; a diagnostic metric must never take an arm down |
| `218e7c3b0` (super `63384314`) | `_plain_gate` reads `to_local()`, never `full_tensor()` | `full_tensor()` is a collective issued from inside the forward, and with an empty local shard on nearly every rank it was a second, independent fault source. The empty shard is what `arch_metrics` wants: its cross-rank sum reconstructs the value and the count sums to one |

## 5. Operational fixes (without these the arm could not run at all)

| problem | fix |
| --- | --- |
| persistent tracker had been overwritten to 35901, whose save is incomplete | resume from the rolling dir instead |
| rolling dir held a **flat** checkpoint (`.metadata` + chunks at the root, from a stop-save) while the loader wants `ckpt_rolling/iter_<N>/` | created `ckpt_rolling/iter_0035996` as a hard-link tree (1024 chunks + 327 MB metadata, no copy) and deleted the flat files; later the tracker moved to the complete `iter_0035999` |
| host008 had 10 exclusions inside its 264 nodes → 254 usable < 256 requested → no 256-node run could be placed | curated exactly 8 exclusions → 256 usable (see below) |
| the node-health preflight exited `rc 86` and could push usable below 256 | run with `NODE_CHECK=off`; the node choice is made by hand from benchmark data |
| v4b was being placed on host007 | priority 150 (v4c is 100), preempt the arm's host007 runs, freeze host007 while settling; host007 unfrozen again because v4c belongs there |
| after the crash the process sits in an uninterruptible kernel call, so the step survives SIGKILL for minutes | the supervisor escalates (`scancel --signal=KILL`); host008 was wedged until the step was reaped |
| small-scale diagnosis was impossible | `acc_debug` caps at 8 nodes (`QOSMaxNodePerJobLimit`) and CPUs (`QOSMaxCpuPerJobLimit`); 8 nodes OOM at parameter init (`tensor_parallel/layers.py: torch.empty`, 1024 MiB requested with 11.2 GiB free); `acc_ehpc` accepts 16 nodes but would start 2026-11-30; `acc_xehpc` is at its 3-job cap. The blocking run had to use the full allocation |

## 6. Node selection for host008

Criterion: exactly 8 exclusions out of 264, so a 256-node run fits. Input was the host's own
benchmark payload (`slurm-48760471.1-node_health_host008.log`): per-GPU matmul TFLOP/s for all 264
nodes plus group bandwidth.

| decision | nodes |
| --- | --- |
| excluded — observed interconnect verdicts | `as02r1b10 as05r5b10 as03r1b06 as07r5b27` |
| excluded — worst minimum-GPU compute (594 / 606 / 623 TFLOP/s vs median 724) | `as04r1b08 as04r1b07 as05r4b19` |
| excluded — IB link flap, `Got non-fatal async event: port active(9)` | `as02r1b08` |
| released — stale option-C entries, near-best compute (rank 253 and 252 of 264) | `as01r2b12 as05r1b31 as01r4b31 as03r5b22` |
| released to hold the count at 8 (compute rank 210) | `as04r1b17` |

Compute and interconnect disagree: `as04r1b17` is a *good* node by compute yet was excluded by a
bandwidth verdict. Interconnect verdicts were kept, because they slow whole collectives.

## 7. Verification (run 1579, iteration 36000)

```
vgate/out 0.01174 | vgate/res -0.008837 | vgate/delta 0.007677
vnorm/q 0.5646 | vnorm/gate 0.427 | vnorm/o 0.3028
mhc_attn/read_share/entropy 0.6856 | mhc_attn/res_diag 0.8361 | mhc_attn/stream_rms/main 2.506
```

The row norms are the scientific point. v4a collapsed to ~5e-4 under MuOwn's magnitude step; v4b
holds 0.30–0.56, so the copy-init plus the learnable gate keeps the branch alive.

Validation at 36000 (all six fixed sets), with v4a for comparison:

| set | v4b | v4a |
| --- | --- | --- |
| all | **1.206720** | 1.224017 |
| web_en | **1.866553** | 1.885757 |
| multilingual | 1.657479 | — |
| code | 0.7857009 | — |
| math | 0.53963 | — |
| reasoning | 0.492022 | — |

`saving checkpoint at iteration 36000` was logged, so rolling checkpointing (every 100) works with
metrics and validation on.

## 8. Remaining risks

- **The eager path costs speed.** Measured on v4c, which now takes it on every step
  (`arch_metrics_interval: 1`): 20.8 s/step vs 19.6 s, about 6%. v4b pays it only every 500 steps.
- **v4c was the more exposed arm, not the less.** With metrics active every step it ran the unfused
  forward on every step, so the fix matters more there. It is applied in `autoexp_v4c` too.
- **The crash wedges a host.** A fault leaves the process in an uninterruptible call; the step must
  be reaped before the host is usable again.
- **The node budget is exactly 256 of 264.** Any further exclusion makes a 256-node run unplaceable.
  Preflight is off, so a degrading node is not removed automatically.
