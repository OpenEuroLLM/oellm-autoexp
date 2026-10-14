# The `defer_embedding_wgrad_compute` bug and the 32B loss divergence

**Last verified:** 2026-09-12
**Production fork observed through:** iteration 72,485
**Overall conclusion:** the gradient bug is demonstrated directly; it is the
high-confidence, dominant cause of the 32B run's late loss divergence, although
this is experimental evidence rather than a mathematical proof.

## 1. Research question

The 32B training loss stopped improving and then began to increase around
iterations 60,000--66,500. The main question is:

> Did `defer_embedding_wgrad_compute: true` cause this late loss divergence?

The answer has two parts:

1. **Is there a software bug? Yes.** Source inspection and a controlled A/B/C
   experiment show that the combination of embedding-weight-gradient deferral
   and overlapping data-parallel gradient reduction produces a different,
   incorrect optimizer gradient.
2. **Did it cause the production loss divergence? Very likely yes.** Starting
   from the same iteration-64,000 checkpoint and changing the flag to `false`
   makes the loss improve immediately and prevents the later rise over more than
   8,000 iterations. A separate frozen-model experiment supports the same
   explanation on held-out data.

It is important to phrase the result precisely. The flag was enabled from the
start of the production run; it did not suddenly switch on at 64k. It therefore
explains the harmful training dynamics, but it does not yet explain why their
visible effect appears only late in training.

## 2. Background

### 2.1 What is the affected parameter?

The final layer of a language model maps each token representation to one logit
per vocabulary item. This document calls it the **LM head** or **output layer**.

The 32B model uses `untie_embeddings_and_output_weights: true`. Its input token
embedding and output layer are therefore different parameters. Despite the flag's
name, the affected production parameter is the LM-head weight, not the input
embedding.

### 2.2 What should data parallelism do?

With data-parallel size \(D\), every rank computes a local gradient \(g_i\). The
optimizer should receive their average:

\[
\bar g = \frac{1}{D}\sum_{i=1}^{D} g_i.
\]

The production run used `D = 128` and global batch size 4,096, so each rank
processed about 32 sequences per optimizer step. Averaging across ranks reduces
the variance of the gradient estimate.

If an optimizer shard instead receives approximately one rank's local gradient,
the estimate may still be unbiased under ordinary sampling assumptions, but its
variance can be about \(D\) times larger. At `D = 128`, that means up to 128 times
the variance, or about \(\sqrt{128} = 11.3\) times the standard deviation. The
actual bug races with an asynchronous collective, so the exact result can be a
timing-dependent mixture rather than a perfectly local gradient.

### 2.3 Why defer the LM-head weight gradient?

Computing the LM-head weight gradient is an expensive matrix multiplication.
Megatron can postpone this computation to the pipeline flush, where it may fill
otherwise idle GPU time. This is a performance optimization and is disabled by
default.

It becomes unsafe here because another optimization,
`overlap_grad_reduce: true`, starts data-parallel communication before all
deferred LM-head gradients have been added.

## 3. The ordering bug

The affected production configuration combines:

- `defer_embedding_wgrad_compute: true`;
- `wgrad_deferral_limit: 0`, meaning all eligible microbatches are deferred;
- `overlap_grad_reduce: true`;
- pipeline parallelism;
- the distributed optimizer; and
- gradient-accumulation fusion.

The relevant order is:

```text
LM-head backward pass saves activations and output gradients
    -> a dummy weight gradient lets the normal DDP hook run
    -> the hook marks the LM-head parameter ready
    -> asynchronous reduce-scatter may start
    -> only later, during pipeline flush, Megatron computes the real LM-head wgrad
    -> the real wgrad is accumulated into main_grad after communication started
```

Consequently, the distributed optimizer does not reliably receive the intended
fully data-parallel-averaged LM-head gradient.

With `overlap_grad_reduce: false`, the reduction is delayed until gradient
finalization. The deferred matrix multiplications are drained first, and the
tested computation is correct. This is why the safe production setting is:

```yaml
overlap_grad_reduce: true
defer_embedding_wgrad_compute: false
```

Keeping gradient-reduction overlap preserves the more valuable communication
optimization while disabling the unsafe deferral.

### Source locations

The ordering can be followed in:

- [`tensor_parallel/layers.py`](../../submodules/Megatron-LM/megatron/core/tensor_parallel/layers.py),
  where the real weight-gradient calculation is deferred and a dummy gradient is
  returned;
- [`distributed_data_parallel.py`](../../submodules/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py)
  and
  [`param_and_grad_buffer.py`](../../submodules/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py),
  where a ready bucket can launch asynchronous reduction; and
- [`pipeline_parallel/schedules.py`](../../submodules/Megatron-LM/megatron/core/pipeline_parallel/schedules.py)
  and [`core/utils.py`](../../submodules/Megatron-LM/megatron/core/utils.py), where
  the deferred LM-head gradient is drained later.

## 4. Evidence 1: controlled one-node reproduction

Slurm job `1760228` tested a small four-layer model on four GPUs. It used PP=2,
DP=2, virtual pipeline stages, an untied LM head, the distributed optimizer, and
the same random seed and mock data in all arms.

| arm | defer LM-head wgrad | overlap grad reduction | result |
|---|---:|---:|---|
| A: reference | off | on | reference trajectory |
| B: suspected bug | on | on | gradient differs before the first update |
| C: ordering control | on | off | exactly matches A at printed precision |

Complete printed results:

| arm | iteration 1: loss / grad norm | iteration 2 | iteration 3 |
|---|---|---|---|
| A | 10.42399 / 0.985 | 10.08325 / 2.011 | 9.727087 / 4.510 |
| B | 10.42399 / **0.994** | **10.09707 / 1.925** | **9.758759 / 3.906** |
| C | 10.42399 / 0.985 | 10.08325 / 2.011 | 9.727087 / 4.510 |

Iteration-1 losses are equal because no optimizer update has occurred yet. The
iteration-1 gradient norm is already different in B, and its loss differs from
iteration 2 onward. A and C match throughout.

This experiment establishes that the bad interaction is **deferral plus
overlapped reduction**, not deferral alone in the tested configuration. It
measures the resulting whole-model gradient norm; the source trace identifies the
LM-head gradient as the late-written parameter.

Raw evidence:

```text
/e/scratch/e-sta-openeurollm/production_training/smoke/defer_wgrad_test/
  defer_wgrad_test.sbatch
  slurm-1760228.log
  run_A.log
  run_B.log
  run_C.log
```

## 5. Evidence 2: intervention in the 32B production run

### 5.1 Experimental design

The intervention resumes from the production run's own persistent
`iter_0064000` checkpoint, including optimizer state. It keeps learning rate
`3e-4` and changes:

```yaml
defer_embedding_wgrad_compute: false
```

The local experiment override is
[`oellm_32b_dense_deferoff_3e-4.yaml`](../../config/experiments/oellm_32b_dense/oellm_32b_dense_deferoff_3e-4.yaml).
The continuation-to-78k override exists in the production checkout used to
submit the jobs.

The comparison uses the original production segment from 64k onward as the
control. It is not a new simultaneous rerun. The audit found:

- the same model, optimizer hyperparameters, learning rate, container, and
  effective data batches;
- per-step loss correlations of `r = 0.75` over 66.5--70k and `r = 0.84` over
  70--71.6k, consistent with the two runs seeing the same changes in batch
  difficulty; and
- no other audited configuration or code difference that should change the
  training mathematics.

The first common logged point, iteration 64,005, is nearly identical before the
new updates have had time to accumulate: 1.544307 with deferral off versus
1.545458 with it on.

### 5.2 Training-loss results

The table below was recomputed from the Slurm logs after de-duplicating restarted
iterations. Full rows are 1,000-iteration means.

| interval | fork: defer off | control: defer on | fork - control |
|---|---:|---:|---:|
| 64--65k | 1.4745 | 1.5440 | -0.0694 |
| 65--66k | 1.4617 | 1.5440 | -0.0824 |
| 66--67k | 1.4545 | 1.5455 | -0.0910 |
| 67--68k | 1.4569 | 1.5526 | -0.0956 |
| 68--69k | 1.4595 | 1.5578 | -0.0983 |
| 69--70k | 1.4532 | 1.5547 | -0.1015 |
| 70--71k | 1.4471 | 1.5591 | -0.1120 |
| 71--72k | 1.4466 | 1.5627 | -0.1161 |
| 72,000--72,485 (partial) | 1.4490 | 1.5682 | -0.1192 |

The effect begins rapidly. These are mean fork-minus-control gaps within the
25-iteration window ending at each listed offset:

| iterations after fork | 25 | 100 | 250 | 500 | 1,000 |
|---|---:|---:|---:|---:|---:|
| 25-iteration-window gap | -0.012 | -0.051 | -0.066 | -0.075 | -0.080 |

The fork is not strictly monotonic because individual batches differ in
difficulty. The relevant observation is that its long-term trend no longer turns
upward and its gap from the control widens in every complete 1,000-iteration bin.
By the latest verified point, it has also run well past the original onset window.

Logs used for this comparison:

```text
/e/project1/e-sta-openeurollm/production_training/
  oellm_32b_dense_prod_dataopt5_gbs4096_lr3e-4/logs/
  oellm_32b_dense_prod_dataopt5_deferoff_3e-4_i64k_seed1234_gbs4096_lr3e-4/logs/
```

## 6. Evidence 3: held-out evaluation and LM-head refitting

Training loss alone could improve because the output layer quickly repairs
itself. To distinguish LM-head damage from damage in the rest of the network, a
second experiment was performed on three checkpoints:

1. freeze every parameter except `output_layer.weight`;
2. evaluate a fixed held-out slice;
3. refit only the LM head for 500 steps with correct gradients, using training
   samples beyond iteration 80,000 that none of the checkpoints had seen; and
4. evaluate the same held-out slice again.

The held-out evaluation contains 80 batches of 512 sequences of length 4,096,
approximately 168 million tokens. Jobs were `1769506`, `1769507`, and `1769508`.

| checkpoint | own-head loss | after head refit | improvement from refit |
|---|---:|---:|---:|
| control 64k | 1.4969 | 1.4491 | 0.0478 |
| control 72k | 1.5206 | 1.4698 | 0.0508 |
| defer-off fork 70k | **1.4006** | **1.3990** | 0.0015 |

Interpretation:

- The fork's head is already close to what this refit procedure can obtain.
- About 0.05 loss in both control checkpoints is directly associated with a
  poorly fitted LM head.
- From control 64k to 72k, the loss after refitting the head rises by about 0.021.
  Thus the late increase cannot be repaired by replacing only the head; it is
  supported by changes in the frozen body of the network.
- Starting from the same 64k weights, the defer-off fork's body improves instead.

The post-refit value is an experimental proxy for what the frozen model body can
support, not an exact mathematical decomposition. A 500-step refit need not find
the global optimum. The two control refit curves remain approximately parallel,
and the fork has almost no remaining head improvement, which makes unfinished
optimization an unlikely explanation for the observed ordering.

Raw evidence:

```text
/e/scratch/e-sta-openeurollm/production_training/smoke/head_refit/
  refit_head.py
  analyze.py
  slurm-1769506.log
  slurm-1769507.log
  slurm-1769508.log
```

## 7. Causal assessment

### 7.1 What is established strongly?

- The affected code ordering exists.
- The suspect flag combination changes the gradient before the first optimizer
  update in a controlled reproduction.
- Disabling gradient-reduction overlap restores the reference result in that
  reproduction.
- At 32B scale, disabling only the suspect optimization from the same checkpoint
  produces an immediate improvement and prevents the historical late rise over
  more than 8,000 iterations.
- Held-out and frozen-body results agree with the training-loss comparison.

Together, these are strong causal evidence. The most useful operational statement
is:

> `defer_embedding_wgrad_compute: true` in combination with overlapped gradient
> reduction was the dominant cause of the observed 32B loss divergence.

### 7.2 Why this is not a 100% proof

No finite training experiment can prove this conclusion with mathematical
certainty. The remaining limitations are:

- there is one production-scale defer-off fork and no second seed;
- the control is the historical production trajectory, not a freshly launched
  concurrent `defer=true` continuation from 64k;
- the fork used a slightly later code revision. The intervening commits were
  audited and appear inactive under these settings, but the original control
  commit is inferred rather than recorded from its old checkout; and
- the head-refit experiment is finite optimization, not a unique analytical
  separation of head and body quality.

These limitations prevent the words "mathematically proven" or "100% certain."
They do not offer a plausible alternative that explains all three independent
results.

### 7.3 What remains unknown?

- Why does a bug present from iteration 0 become visibly harmful only around
  60--66k?
- Through what detailed mechanism does a noisy LM head degrade the transformer
  body? One plausible explanation is that upstream layers must learn through a
  head whose parameters move noisily, but this has not been isolated.
- How much damage from the first 64,000 affected updates remains in the recovered
  fork?

## 8. Version and configuration provenance

Three code lines matter:

1. **Megatron Core v0.16.0.** The official release commit
   `3bec9aa97dda898d16ff5a89bac0ed2b6682b172` has the affected ordering. The local
   `origin/v0.16.0-openeurollm` branch descends from it, and its OpenEuroLLM
   commits do not modify the relevant files. Therefore the issue is present in
   the v0.16.0 code, not introduced by those local patches.
2. **The currently checked-out submodule.** It is
   `8e0c0f4bc3ec5007b2e4fb63265b077a82eff31d`, described as
   `v0.15-openeurollm-10-g8e0c0f4bc`. Job `1760228` reproduces the interaction on
   this fork.
3. **The actual 32B production line.** The production control is inferred to have
   used `7f6ea8d80`; the defer-off fork used `e6d2aa72` on the later 0.19-based
   OpenEuroLLM fork. The relevant ordering is also present there.

The flag is **not a Megatron default**. Megatron v0.16.0 declares it `False`, and
[`config/backend/megatron/base_defaults.yaml`](../../config/backend/megatron/base_defaults.yaml)
also defaults to `false`. This project enabled it explicitly for the 32B scaling
configuration in commit `f27f111` on 2026-08-14 and copied it into the production
configuration in commit `59b3caa` on 2026-08-21. Commit `136dbc6` disabled it for
the v2 configuration on 2026-09-11.

The production runtime dump from job `1537344` confirms:

| setting | value |
|---|---:|
| defer embedding wgrad | true |
| deferral limit | 0 |
| overlap grad reduction | true |
| distributed optimizer | true |
| gradient-accumulation fusion | true |
| untied input/output weights | true |
| TP / PP / VPP / DP | 4 / 4 / 4 / 128 |
| global / micro batch size | 4096 / 2 |

## 9. Recommended actions

1. Keep `defer_embedding_wgrad_compute: false` whenever
   `overlap_grad_reduce: true` on these Megatron revisions.
2. Add a startup assertion that rejects the unsafe combination until the
   reduction ordering is fixed and covered by a regression test.
3. Add an upstream-style test containing pipeline parallelism, an untied output
   layer, the distributed optimizer, deferral, and overlapped reduction. Compare
   the actual LM-head gradient or one optimizer update against a non-deferred
   reference.
4. Prefer a checkpoint from the defer-off fork over a later affected control
   checkpoint. Refitting only the output layer does not remove the measured
   frozen-body degradation.
5. Treat learning-rate conclusions drawn from the affected arms cautiously. A
   lower learning rate reduced the size of noisy updates and looked like a partial
   fix, but it did not remove the gradient-ordering bug.
