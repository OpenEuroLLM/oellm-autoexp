# v4b plan: bringing the V attention in without a first-step transient

Status: plan only. No main config is edited. The running arm `v4a_v6_pin` is not
touched. v4b is a contingency if v4a does not catch up.

## 1. Is the inserted V layer a no-op? No.

The mHC sublayer write is:

```
streams' = H_res @ streams + H_post^T f(x)
```

with

```
H_post = 2 * sigmoid(a_post * post_raw + b_post)
H_res  = Sinkhorn(a_res * res_raw + b_res)        # doubly stochastic
```

At the V init (`phi = 0`, `b_post = 0`, `b_res = 5 I`, `alpha_* = 0.01`,
`o_proj = 0`):

- `post_raw = 0` and `res_raw = 0`, because both come from `flat @ phi`.
- `H_post = 2 * sigmoid(0) = 1.0` exactly. The branch coefficient is exact.
- `f(x) = 0` exactly, because `o_proj = 0`. The branch term is exact.
- `H_res = Sinkhorn(5 I) = 0.9802 I + 0.0066 (J - I)`. **Not the identity.**

`H_res` is doubly stochastic, so the stream magnitude is preserved. But each
inserted layer applies one extra near-identity **cross-stream mixing**: every
stream takes 0.66 % of each of the other three streams. The parent has no layer
at that position, so the parent has no such mixing there. Six insertions add six
extra mixings.

So the correct statement is:

- the **branch** is an exact no-op;
- the **residual path** is not, and it is a small, real functional change.

This matches the measurements already on record:

- 35900 parity, v4 vs v4a: 0.1-1.1 pp on CoQA / SQuAD v2 / Belebele. Not bf16 noise.
- first logged step: `lm loss 1.103468` against the 1.1027 reference, delta 8e-4.

`verify_v_surgery.py` reports `PASS identity` because it checks tensors: zero
`o_proj`, unit norms, zero `phi`, `b_res = 5 I`. It does not check the function of
`H_res`. That is why the claim was corrected later in
`docs/recall_surgery_ambitious.md` (section on the residual rewrite).

## 2. Two separate defects

D1 - static, present at step 0
: the extra `H_res` mixing, six times. It is a small offset, not a transient.

D2 - transient, grows over about 100 steps
: the branch is a cold start. `o_proj = 0` gives zero gradient to q and gate at
step 0, while q and gate are random at 0.02. Only `o_proj` grows, at the full base
LR (8.5e-4). The branch therefore goes from silence to injecting **noise** over
about 100 steps, and the model has to recover from that injection.

The recall evals at +100 steps sit inside D2, so they measure the transient, not a
permanent loss of quality. That is a hypothesis to test, not a conclusion.

## 3. Options

| option | exact at step 0 | removes D2 | cost | risk |
| --- | --- | --- | --- | --- |
| A. copy-init branch + output gate `g_v = 0` (+ residual gate `rho = 0`) | yes, bit-exact | yes | surgery + 2 scalars per layer + export support | medium |
| B. per-group LR warmup on the six V layers only | no | partly | config only | low |
| C. staged insertion, 2 layers per restart | yes | partly | 3 restarts | low |
| D. serial branch form (`ckda_branch_serial: true`), no new depth | yes | yes | config + existing branch code | low |

### A. Copy-init branch plus a zero output gate

Replace the random `qkv` and the zero `o_proj` with a copy of the source attention
layer, exactly as `add_ckda_branch.py --init attn` already does:

- copy q / k / v / o and the qk-norm from the source layer;
- fold the source input-norm gain into the `linear_qkv` columns
  (`W' = W diag(g_attn / g_layer)`), because the branch reads the layer's normed input;
- add one scalar gate per V layer: `f(x) = g_v * attention(x)`, with `g_v = 0`.

Then at step 0 the branch term is exact, and the branch is **already a working
attention** the moment `g_v` moves. That is the argument the campaign already makes
for the additive branch: 300 steps are spent adapting a working path instead of
bootstrapping a random one.

Also gate the residual path:

```
streams' = (1 - rho) * streams + rho * (H_res @ streams) + H_post^T f(x)
```

with `rho = 0` at init. This makes D1 bit-exact. `rho` then learns the mixing
instead of starting at 0.9802 I by construction. `rho` is a scalar per inserted
layer, so the cost is negligible and the export needs one extra scalar per V layer.

Ceiling: `rho = 0` also blocks the gradient to `b_res` / `phi` of that layer at
step 0. That is acceptable, because those parameters have no job until the layer
carries signal. Upgrade trigger: if `rho` stays below 1e-3 after 300 steps, the
gate is mis-scaled, not the init.

### B. Per-group LR warmup

The repo already has per-group LR multipliers (`attention_lr_mult`,
`residual_mixer_lr_mult`, `engram_lr_mult`). Add a ramp for the six V layers:
`v_lr_mult` 0 -> 1 over 200-500 steps, covering q, gate, `o_proj` and the norms.
Add a short global re-warmup at the restart, for example 50 steps from 0.1x base.

B alone lowers the spike amplitude. It does not remove the random-branch
bootstrap, so B is a mitigation, not a fix.

### C. Staged insertion

Insert two layers, train, insert two more, and so on. Each restart has a smaller
shock. Cost is three restarts and three segment boundaries. Use this only if A is
unavailable.

### D. Serial branch, no new depth

`ckda_branch_serial: true` adds the attention inside the existing CKDA layer. It
adds no layer, so it adds **no mHC write at all**, and it is exact at `gamma = 0`
with a copy-init branch. If the objective is recall and not depth, this is the
least invasive route to the same attention capacity, and it is already implemented.

## 4. Recommended v4b

Use **A + B together**, and keep D as the fallback if A needs more surgery than the
schedule allows.

Reason: A removes both D1 and D2 by construction. B then controls how fast the
branch share grows, which A alone does not set. One without the other leaves either
a cold start (A alone with a fast gate ramp) or a random branch (B alone).

Proposed spec for the v4b arm, to be written as a new config file, not by editing
`v4a_v6_pin_mn5.yaml`:

- start from the same V6 35900 checkpoint, new output directory;
- new layer init: copy-init branch, `g_v = 0`, `rho = 0`, `b_post = 0`;
- gate ramp: `g_v` share from 0 to the calibrated target over 200 steps, using the
  existing `calib = RMS(main) / RMS(branch)` measured on the first forward;
- `v_lr_mult` ramp 0 -> 1 over 300 steps;
- everything else unchanged: same data order, same saves, same validation, same
  Engram pin.

## 5. Acceptance checks

Run these in order. Each one must pass before the next is trusted.

1. Step-0 identity: first logged loss within **1e-4** of the parent, not 8e-4.
2. Functional residual check: on a forward pass, a V layer with `rho = 0` must
   return the input streams bit-exactly. Do not accept a tensor inspection.
3. Transient envelope: max loss over steps +1..+100 within +0.01 of the parent, and
   no single step above +0.02.
4. Recall at +100 and +300 steps: within noise of the parent, not below it.
5. Grad norm at steps 1-10 within the parent's band.

## 6. Decision rule

- `v4a_v6_pin` continues. No config edit and no Slurm change.
- Decide at the **36900** save, using the 36400 / 36900 recall readouts against the
  35900 paired baseline.
- Run v4b only if v4a is still below the parent on recall at that point. A v4b run
  costs one restart from 35900 plus a 24 h segment, so it must be justified by a
  measured gap, not by the step-0 spike alone.

## 7. Zero-cost diagnostics before any new run

These need no new allocation and no restart.

1. Compare the current run's loss at 35901-36000 against the parent `v4a_v6` run at
   the same steps. That separates D2 from normal step-to-step noise.
2. Read `H_res` from the exported 35900 checkpoint for layers 10 / 27 / 44 / 62 /
   79 / 96 and confirm `0.9802 I + 0.0066 (J - I)`. This confirms D1 directly.
3. Reuse the existing 35900 parity evals as the D1 baseline; do not re-run them.

## 8. Not doing

- No edit to `config/experiments/korbi/recall_surgery/v4a_v6_pin_mn5.yaml`.
- No change to the running job or its Slurm allocation.
- No v4b submission until the 36900 readout exists.

## 9. Measured status of the implemented gate (2026-11-17, eval side)

What is actually in the tree is the **schedule gate only** (`v4b_gate_mn5.yaml`:
`v_block_gate_ramp: 300`, `v_block_gate_start_iteration: 35900`), not option A and not
option B. Read from `transformer_layer.py:1142`:

```python
written = mixer.write(streams, output, h_post, h_res, ...)
gate = self._v_block_gate(kind)          # r = clamp((t - 35900) / ramp, 0, 1); None once r >= 1
return torch.lerp(streams, written, gate)
```

Consequences that matter for "substantial impact":

- **D1 is fixed bit-exactly.** At `r = 0` the layer returns the incoming streams, so the
  residual mixing and the attention write are both absent, and the 35900 baseline becomes a
  true parity check (the 0.1-1.1 pp offset of the v4a pair should disappear).
- **D2 is delayed, not removed.** `r` multiplies the gradient into every V parameter, so at
  `r = 0` those gradients are exactly zero and the branch learns nothing for the first steps.
  After that it is still the same random-init branch with `o_proj = 0`. By 36400 the branch has
  had ~200 full-gate steps plus a 300-step ramp average, i.e. *less* effective learning than
  v4a at the same step.
- **A checkpoint saved inside `[35900, 36200)` must not be exported.** `_v_block_gate` returns
  `None` when there is no training loop, so the export always writes whole: a mid-ramp
  checkpoint would silently export a different function than it trained. Export at >= 36200.

Still missing for impact, in priority order:

1. copy-init branch (q / k / v / o + qk-norm from the source attention layer, source input-norm
   gain folded into `linear_qkv`), so the branch is functional when the gate opens;
2. a **learnable** zero-init output gate `g_v` (`dL/dg_v = <delta, attn_out> != 0` at 0) instead
   of the schedule, which has no gradient at all;
3. `rho = 0` on the residual path, so D1 is exact *and* learnable;
4. an LR multiplier >= 1 for the V tensors, and a ramp <= 100 steps if a schedule is kept.

Acceptance signal, already instrumented by `_record_v_block_impact` (`vgate/r`, `vgate/delta`
= `||write - streams|| / ||streams||` per layer), no eval needed:

| `vgate/delta` | reading |
| --- | --- |
| <= 1e-4 | inert (v4a's regime: `o_proj` 30-80x below a trained attention layer) |
| ~1e-3 | noticeable |
| >= 1e-2 | substantial |

Calibration: a trained attention layer moves ~2-9 % of the stream (`o_proj` RMS 2.9e-3, output
~0.09 against a stream of order 1-4), so six V layers at trained scale would move 10-50 % of it.
The ceiling is set by the init and the gradient path, not by the architecture.
