#!/usr/bin/env python3
"""CPU-only tests for the pure logic in megatron/training/diagnostics.py.

These cover the parts that are wrong SILENTLY rather than loudly: slot
assignment (a collision shows up as two layers averaged together, not as a
crash) and the clip streak counter (an off-by-one shows up as a wrong alarm
threshold). The parts that need real parallel groups are covered by
config/experiments/korbi/diagnostics_smoke_130M.yaml instead.

    PYTHONPATH=submodules/Megatron-LM:. python3 scripts/korbi/test_diagnostics_logic.py
"""

import sys
import types

sys.path.insert(0, "submodules/Megatron-LM")

from megatron.training.diagnostics import (  # noqa: E402
    GAIN_FAMILIES,
    GRAD_EXTRAS,
    TrainingDiagnostics,
    _classify_gain,
    _classify_grad_bucket,
)

FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


def fake_args(**kw):
    a = types.SimpleNamespace(
        num_layers=18,
        diagnostics_interval=0,
        diag_norm_gains=False,
        diag_layer_grad_norms=False,
        diag_nonfinite=False,
        diag_activations=False,
        diag_clip_events=True,
        clip_grad=1.0,
        use_distributed_optimizer=True,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


# --- family classification --------------------------------------------------
# The TE-fused spellings are the ones that matter: matching only "layernorm"
# would silently drop the two biggest families into `other_norm`.
check(
    "fused qkv gain",
    _classify_gain("decoder.layers.3.self_attention.linear_qkv.layer_norm_weight"),
    "input_norm",
)
check(
    "fused fc1 gain",
    _classify_gain("decoder.layers.3.mlp.linear_fc1.layer_norm_weight"),
    "pre_mlp_norm",
)
check("q gain", _classify_gain("decoder.layers.3.self_attention.q_layernorm.weight"), "q_norm")
check("k gain", _classify_gain("decoder.layers.3.self_attention.k_layernorm.weight"), "k_norm")
check("final gain", _classify_gain("decoder.final_layernorm.weight"), "other_norm")
check("not a gain", _classify_gain("decoder.layers.3.mlp.linear_fc2.weight"), None)
check("embedding is not a gain", _classify_gain("embedding.word_embeddings.weight"), None)

check("embed bucket", _classify_grad_bucket("embedding.word_embeddings.weight"), "embedding")
check("head bucket", _classify_grad_bucket("output_layer.weight"), "output_layer")
check("final bucket", _classify_grad_bucket("decoder.final_layernorm.weight"), "final_norm")

# --- slot assignment --------------------------------------------------------
d = TrainingDiagnostics(fake_args())
n = d.num_layers

# Every (family, layer) pair must map to its own slot: a collision would average
# two different tensors together and look like a plausible number.
seen = {}
for fam in GAIN_FAMILIES:
    for li in range(n):
        s = d._gain_slot(fam, li)
        if s in seen:
            FAILURES.append(f"gain slot collision: {(fam, li)} and {seen[s]} both -> {s}")
        seen[s] = (fam, li)
check("gain slot count", len(seen), len(GAIN_FAMILIES) * n)
check("final_norm slot is separate", d._gain_slot("other_norm", None) in seen, False)
check("final_norm slot value", d._gain_slot("other_norm", None), len(GAIN_FAMILIES) * n)

gslots = {d._grad_slot(f"decoder.layers.{li}.mlp.linear_fc1.weight", li) for li in range(n)}
check("per-layer grad slots distinct", len(gslots), n)
check("layer grad slots are the first n", sorted(gslots), list(range(n)))
for i, extra in enumerate(GRAD_EXTRAS):
    name = {
        "embedding": "embedding.word_embeddings.weight",
        "output_layer": "output_layer.weight",
        "final_norm": "decoder.final_layernorm.weight",
        "other": "some.other.weight",
    }[extra]
    check(f"extra slot {extra}", d._grad_slot(name, None), n + i)

# A layer index beyond num_layers must clamp, not index out of the buffer.
check("layer index clamps", d._grad_slot("decoder.layers.99.mlp.linear_fc1.weight", 99), n - 1)

# --- clip streak ------------------------------------------------------------
# grad_norm > clip_grad means the clip fired. A streak must count CONSECUTIVE
# fires and reset to 0 on the first step that does not fire.
d = TrainingDiagnostics(fake_args(clip_grad=1.0))
for gn in (0.5, 0.5):
    d.observe_clip(gn)
check("no streak while under the ceiling", d.clip_streak, 0)
for gn in (2.0, 3.0, 4.0):
    d.observe_clip(gn)
check("streak counts consecutive fires", d.clip_streak, 3)
check("max streak recorded", d._clip_max_streak, 3)
d.observe_clip(0.5)
check("streak resets on a clean step", d.clip_streak, 0)
check("max streak survives the reset", d._clip_max_streak, 3)
d.observe_clip(9.0)
check("streak restarts at 1", d.clip_streak, 1)

# The clip coefficient is what actually scales the gradient.
d2 = TrainingDiagnostics(fake_args(clip_grad=1.0))
d2.observe_clip(4.0)
check("coeff_min ~ clip_grad/grad_norm", round(d2._clip_coeff_min, 6), round(1.0 / (4.0 + 1e-6), 6))
check("grad-norm max tracked", d2._gn_max, 4.0)
check("grad-norm min tracked", d2._gn_min, 4.0)

# A NaN grad norm must not poison the accumulators or the streak.
d3 = TrainingDiagnostics(fake_args(clip_grad=1.0))
d3.observe_clip(2.0)
d3.observe_clip(float("nan"))
check("NaN grad norm is ignored", d3._clip_steps, 1)
check("NaN does not extend the streak", d3.clip_streak, 1)

# clip_grad = 0 means clipping is off; nothing should ever read as fired.
d4 = TrainingDiagnostics(fake_args(clip_grad=0.0))
d4.observe_clip(1e9)
check("no clip when clip_grad is 0", d4.clip_streak, 0)

# --- gating -----------------------------------------------------------------
off = TrainingDiagnostics(fake_args(diagnostics_interval=0, diag_norm_gains=True))
check("interval 0 disables collectors", off.want_gains, False)
check("interval 0 leaves clip events alone", off.want_clip, True)
on = TrainingDiagnostics(fake_args(diagnostics_interval=10, diag_norm_gains=True))
check("interval enables collectors", on.want_gains, True)
on.begin_step(100)
check("diag iteration on a multiple", on._is_diag_iter, True)
on.begin_step(105)
check("not a diag iteration otherwise", on._is_diag_iter, False)
check("emit is empty with nothing collected", on.emit(105, None, None), {})

if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print("  " + f)
    sys.exit(1)
print("all diagnostics logic tests passed")
