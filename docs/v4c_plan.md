# v4c plan: copy-init V layers brought in by a short schedule ramp

Status: implemented (surgery config, arm config, smoke, exporter guard). Not launched — the running
side is delegated.

v4c is the third arm on the same 35900 source and the same copy-init surgery as v4b
(`docs/v4b_plan.md`), with one variable: the branch is brought in by a **100-step schedule ramp**
instead of the two learnable gates.

## 1. Why not keep v4b's learnable gates

Measured on the v4b run (`docs/recall_surgery_ambitious.md` §22.19, `docs/v4b_plan.md` §9):

- `out_gate` / `res_gate` are one-element parameters, so Adam moves them by ~`lr = 8.5e-4` per step
  whatever the gradient's size: `g_v` reads ~0.085 at 35999, ~0.42 at 36400, ~0.85 at 36900, and 1.0
  only around 37400.
- the branch matrices' gradient is `proportional to g_v`, so their cumulative exposure is
  `integral g_v dt = lr n^2 / 2` — about 100 effective steps by 36400 and 400 by 36900. That is not
  enough for weights trained at depth 0 to adapt to depth 10.
- the cost of an unadapted branch is measurable and grows with the gate: **+0.0001 nats** at
  `g_v = 0.085`, extrapolating to ~+0.001 nats at full weight (matched-iteration loss, 99 iterations,
  ±1e-4 resolution).

So v4b is correct but cannot test the branch inside any window we can afford. v4c buys that window by
opening the branch deterministically in 100 steps.

## 2. Design

| file | what it does |
| --- | --- |
| `config/experiments/korbi/recall_surgery/v4c_surgery_mn5.yaml` | the copy-init surgery into its own tree `runs/v4c_src` (same script as v4b's, so v4c does not depend on v4b's source tree) |
| `config/experiments/korbi/recall_surgery/v4c_ramp_mn5.yaml` | the arm: `v_block_gate_ramp: 100` from 35900, learnable gate off, loads `runs/v4c_src/ckpt` |
| `config/experiments/korbi/recall_surgery/v4c_smoke_mn5.yaml` | 6 steps, 16 nodes, arch metrics and parameter norms on every step, with the acceptance numbers |

The layer writes `streams + r(t) * (mhc_write(...) - streams)` with `r = clamp((t - 35900) / 100, 0, 1)`:

- `r = 0` at 35900: bit-exact parent (no attention write, no `h_res` mixing);
- `r = 1` from 36000: the branch writes whole, and the residual mixing comes in with it (0.0008 nats
  at full, i.e. free);
- the copied matrices get gradient `proportional to r` from step 1, so they adapt immediately instead
  of after the gate has crawled open.

Saves at 35905 / 35999 / 36000 / 36100 / 36400 / 36900, rolling every 100. The run is open-ended to
the parent's `train_iters`, like v4b.

**Eval discipline:** never export a checkpoint inside `[35900, 36000)`. The exporter derives that
window from the run's own `v_block_gate_ramp` / `v_block_gate_start_iteration` (commit `601a9fe3`);
for learnable gates it instead refuses per layer when `|g_v| < 0.05`, since a nearly closed branch is
an almost-v4 readout.

## 3. The gate gain, deferred

The follow-up, worth building only if the ramp shows the branch helping, is a constant gain on the
gate parameter: `g_v = 8 * raw` with `raw` zero-init, so one Adam step moves `g_v` by `8 * lr` and the
gate opens in ~150 steps while keeping self-regulation. It needs three things, which is why it is not
in this arm:

1. a Megatron config field (`v_gate_gain`, default 1.0, so v4b's stored gate values keep their meaning);
2. the autoexp schema and the generated CLI metadata for that field;
3. the same gain in the exporter's `g_v` fold;

each with a unit test. A separate LR multiplier for the branch matrices is *not* needed: raising the
gate multiplies those gradients by the same factor.

## 4. Open questions this arm answers

- Does the copied branch adapt to its new depth once it carries full weight from 36000? Signal: the
  matched-iteration loss (±1e-4 nats), not the evals.
- Does the residual mixing cost anything at `r = 1`? Expected no (0.0008 nats).
- Does the branch show up behaviourally? RULER 4k–32k and the recall suite are the no-regression
  checks; the `niah_multiquery` rows are the ones that separated the arms once before
  (`docs/recall_surgery_ambitious.md` §22.20).

## 5. Not doing

- No edit to `v4a_v6_pin_mn5.yaml` / `v4b_learned_mn5.yaml`; v4c only overrides what it names.
- No new surgery code: the copy-init surgery is reused as is.
- No launch here; the run order is surgery (1 node, ~40 min) → smoke (16 nodes, ~16 min) → arm.
