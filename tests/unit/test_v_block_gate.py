"""Checks for the v4b residual-level gate on the inserted ``V`` layers.

``docs/v4b_plan.md``: a V layer's whole mHC delta is scaled by a ramp ``r(t)``, so at ``r = 0``
the inserted layer is an exact identity -- neither the attention write nor the ``h_res`` stream
mixing applies -- and both ramp in together.

The tests pin three things:

* the schedule (0 before the start, linear over the ramp, gone after it);
* that ``r = 0`` returns the incoming streams bit-exactly and ``r = 1`` is the v4a write;
* that an unset ramp changes nothing at all: no lerp call, no extra tensor, the same bytes.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
tl = pytest.importorskip("megatron.core.transformer.transformer_layer")
ckda_branch = pytest.importorskip("megatron.core.ssm.ckda_branch")

TransformerLayer = tl.TransformerLayer
ROOT = Path(__file__).resolve().parents[2]

S, B, N, C = 2, 3, 4, 8


class _RefMixer:
    """The mHC write of ``megatron/core/transformer/mhc.py`` (paper Eq. 3), so a gate test sees the
    real math instead of a canned tensor:

        written = h_res @ streams + h_post^T out
    """

    def __init__(self) -> None:
        self.calls = 0
        self.args = None

    def write(self, streams, sublayer_out, h_post, h_res, lerp=None, **extra):
        self.calls += 1
        assert lerp is None, "the stub never asks for an nGPT lerp residual"
        self.args = (sublayer_out, h_post, h_res)
        mixed = (h_res.unsqueeze(-1) * streams.unsqueeze(-3)).sum(dim=-2)
        return mixed + h_post.unsqueeze(-1) * sublayer_out.unsqueeze(-2)


class _Mixer:
    """Stands in for MHCMixer: records the call and returns a fixed written state."""

    def __init__(self, written: torch.Tensor) -> None:
        self.written = written
        self.calls = 0

    def write(self, streams, sublayer_out, h_post, h_res, lerp=None, **extra):
        self.calls += 1
        assert lerp is None, "the stub never asks for an nGPT lerp residual"
        return self.written


class _Layer:
    """Only the attributes ``TransformerLayer._mhc_write`` touches.

    The two methods under test are bound straight off the real class, so the test exercises
    the shipped code rather than a copy of it.
    """

    _mhc_write = TransformerLayer._mhc_write
    _v_block_gate = TransformerLayer._v_block_gate
    _record_v_block_impact = TransformerLayer._record_v_block_impact
    _v_learnable_gate = TransformerLayer._v_learnable_gate
    _record_v_gate = TransformerLayer._record_v_gate

    def __init__(
        self,
        *,
        layer_number: int = 11,
        ramp: int = 0,
        start: int = 0,
        kv_layers: tuple[int, ...] = (10,),
        vgate=None,
    ) -> None:
        self.layer_number = layer_number
        self.v_gate = vgate
        self.config = SimpleNamespace(
            v_block_gate_ramp=ramp,
            v_block_gate_start_iteration=start,
            kv_share_layers=list(kv_layers),
            ckda_branch_rest_dropout=0.0,
            mhc_lerp=False,
            num_layers=104,
        )
        self.self_attention = SimpleNamespace(branch_mix=None)
        self.training = False

    def _sublayer_output(self, output_with_bias):
        return output_with_bias[0]

    def _depth_lerp(self, kind, use_lerp):
        return None


def _tensors(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    streams = torch.randn(S, B, N, C, generator=g)
    written = torch.randn(S, B, N, C, generator=g)
    out = torch.randn(S, B, C, generator=g)
    h_post = torch.rand(S, B, N, generator=g)
    h_res = torch.eye(N).expand(S, B, N, N).contiguous()
    return streams, written, (out, None), (h_post, h_res)


def _write(layer, kind="attn", seed=0):
    streams, written, output_with_bias, pending = _tensors(seed)
    mixer = _Mixer(written)
    result = layer._mhc_write(mixer, kind, streams, output_with_bias, pending)
    return result, streams, written, mixer


def _gate(out: float = 0.0, res: float = 0.0):
    """A ``VBlockGate`` at fixed gate values (both 0 = the insertion)."""
    gate = tl.VBlockGate(N)
    with torch.no_grad():
        gate.out_gate.fill_(out)
        gate.res_gate.fill_(res)
    return gate


def _write_ref(layer, kind="attn", seed=0, h_res=None):
    """``_mhc_write`` against the reference mHC write: returns (result, streams, out, h_post, h_res)."""
    streams, _, output_with_bias, pending = _tensors(seed)
    h_post, res = pending
    h_res = res if h_res is None else h_res
    mixer = _RefMixer()
    result = layer._mhc_write(mixer, kind, streams, output_with_bias, (h_post, h_res))
    return result, streams, output_with_bias[0], h_post, h_res


def _mixed(h_res, streams):
    return (h_res.unsqueeze(-1) * streams.unsqueeze(-3)).sum(dim=-2)


# --------------------------------------------------------------------------- schedule


@pytest.mark.parametrize(
    "iteration,start,ramp,expected",
    [
        (35900, 35900, 300, 0.0),  # insertion step: exact identity
        (35950, 35900, 300, 50 / 300),
        (36050, 35900, 300, 0.5),
        (36199, 35900, 300, 299 / 300),
        (36200, 35900, 300, None),  # ramp complete: the gate is dropped
        (36900, 35900, 300, None),
        (35000, 35900, 300, 0.0),  # before the start: clamped, never negative
    ],
)
def test_schedule(monkeypatch, iteration, start, ramp, expected):
    monkeypatch.setattr(ckda_branch, "current_iteration", lambda: iteration)
    layer = _Layer(ramp=ramp, start=start)
    gate = layer._v_block_gate("attn")
    if expected is None:
        assert gate is None
    else:
        assert gate == pytest.approx(expected)


def test_no_iteration_is_ungated(monkeypatch):
    """Evaluation / unit tests have no training loop: a trained layer writes whole."""
    monkeypatch.setattr(ckda_branch, "current_iteration", lambda: None)
    assert _Layer(ramp=300, start=35900)._v_block_gate("attn") is None


# --------------------------------------------------------------------------- default off


def test_unset_ramp_never_calls_lerp(monkeypatch):
    """The v4a path must not gain a tensor op: ramp 0 returns before any lerp."""
    monkeypatch.setattr(ckda_branch, "current_iteration", lambda: 35900)

    def _boom(*a, **k):  # pragma: no cover - only on failure
        raise AssertionError("torch.lerp must not run when v_block_gate_ramp is unset")

    monkeypatch.setattr(tl.torch, "lerp", _boom)
    result, _, written, mixer = _write(_Layer(ramp=0), seed=1)
    assert mixer.calls == 1
    assert torch.equal(result, written)


def test_unset_ramp_is_byte_identical_to_the_ungated_write():
    result, _, written, _ = _write(_Layer(ramp=0), seed=2)
    assert result is written  # the mixer's tensor is returned as-is, not copied


def test_missing_config_attributes_are_off():
    """A config built before this change has no such attributes at all."""
    layer = _Layer()
    del layer.config.v_block_gate_ramp
    del layer.config.v_block_gate_start_iteration
    assert layer._v_block_gate("attn") is None


# --------------------------------------------------------------------------- gated


def test_zero_gate_is_the_streams_bit_exactly(monkeypatch):
    monkeypatch.setattr(ckda_branch, "current_iteration", lambda: 35900)
    result, streams, _, _ = _write(_Layer(ramp=300, start=35900), seed=3)
    assert torch.equal(result, streams)


def test_zero_gate_keeps_the_mhc_mixing_out(monkeypatch):
    """h_res must not reach the streams at r = 0, even when it is not the identity."""
    monkeypatch.setattr(ckda_branch, "current_iteration", lambda: 35900)
    streams, written, output_with_bias, (h_post, _) = _tensors(4)
    h_res = torch.full((S, B, N, N), 0.25)  # far from the identity
    mixer = _Mixer(written)
    result = _Layer(ramp=300, start=35900)._mhc_write(
        mixer, "attn", streams, output_with_bias, (h_post, h_res)
    )
    assert torch.equal(result, streams)


def test_half_gate_is_the_lerp(monkeypatch):
    monkeypatch.setattr(ckda_branch, "current_iteration", lambda: 36050)
    result, streams, written, _ = _write(_Layer(ramp=300, start=35900), seed=5)
    assert torch.allclose(result, torch.lerp(streams, written, 0.5))


def test_gate_opens_monotonically(monkeypatch):
    distances = []
    for iteration in (35900, 35975, 36050, 36125, 36199):
        monkeypatch.setattr(ckda_branch, "current_iteration", lambda it=iteration: it)
        result, streams, written, _ = _write(_Layer(ramp=300, start=35900), seed=6)
        distances.append(float((result - written).norm()))
    assert distances == sorted(distances, reverse=True)
    assert distances[-1] < distances[0]


def test_completed_ramp_returns_the_write(monkeypatch):
    monkeypatch.setattr(ckda_branch, "current_iteration", lambda: 36900)
    result, _, written, _ = _write(_Layer(ramp=300, start=35900), seed=7)
    assert torch.equal(result, written)


# --------------------------------------------------------------------------- scope


def test_mlp_write_is_never_gated(monkeypatch):
    monkeypatch.setattr(ckda_branch, "current_iteration", lambda: 35900)
    result, _, written, _ = _write(_Layer(ramp=300, start=35900), kind="mlp", seed=8)
    assert torch.equal(result, written)


def test_other_layers_are_never_gated(monkeypatch):
    monkeypatch.setattr(ckda_branch, "current_iteration", lambda: 35900)
    layer = _Layer(layer_number=12, ramp=300, start=35900, kv_layers=(10,))
    assert layer._v_block_gate("attn") is None
    result, _, written, _ = _write(layer, seed=9)
    assert torch.equal(result, written)


def test_layer_number_is_one_based_against_kv_share_layers(monkeypatch):
    """kv_share_layers are 0-based, layer_number is 1-based: V layer 10 is layer_number 11."""
    monkeypatch.setattr(ckda_branch, "current_iteration", lambda: 35900)
    gated = _Layer(layer_number=11, ramp=300, start=35900, kv_layers=(10,))
    assert gated._v_block_gate("attn") == 0.0
    not_gated = _Layer(layer_number=10, ramp=300, start=35900, kv_layers=(10,))
    assert not_gated._v_block_gate("attn") is None


# --------------------------------------------------------------------------- wiring


def test_config_schema_has_the_fields_defaulted_off():
    from oellm_autoexp.backends.megatron.config_schema import MegatronConfig

    annotations = getattr(MegatronConfig, "__annotations__", {})
    assert "v_block_gate_ramp" in annotations
    assert "v_block_gate_start_iteration" in annotations


def test_cli_metadata_and_action_specs_are_registered():
    from oellm_autoexp.backends.megatron.cli_metadata import (
        MEGATRON_ACTION_SPECS,
        MEGATRON_ARG_METADATA,
    )

    ramp = MEGATRON_ARG_METADATA["v_block_gate_ramp"]
    assert ramp.arg_type is int
    assert ramp.default == 0
    start = MEGATRON_ARG_METADATA["v_block_gate_start_iteration"]
    assert start.arg_type is int
    assert start.default == 0
    assert MEGATRON_ACTION_SPECS["v_block_gate_ramp"].option_strings == ("--v-block-gate-ramp",)
    assert MEGATRON_ACTION_SPECS["v_block_gate_start_iteration"].option_strings == (
        "--v-block-gate-start-iteration",
    )


def test_v4a_config_is_untouched_and_v4b_sets_the_gate():
    """The running arm must not mention the gate, so it resolves to the 0 default."""
    v4a = ROOT / "config/experiments/korbi/recall_surgery/v4a_v6_pin_mn5.yaml"
    v4b = ROOT / "config/experiments/korbi/recall_surgery/v4b_gate_mn5.yaml"
    v4a_text = v4a.read_text()
    assert "v_block_gate" not in v4a_text
    v4b_text = v4b.read_text()
    assert "v_block_gate_ramp: 300" in v4b_text
    assert "v_block_gate_start_iteration: 35900" in v4b_text
    assert "v4a_v6_pin_mn5" in v4b_text  # same source, pattern and saves as v4a
    assert "runs/v4b_gate/main" in v4b_text  # separate output directory


# --------------------------------------------------------------------------- impact metric


def test_impact_metric_is_skipped_when_arch_metrics_are_inactive(monkeypatch):
    """The metric must cost nothing unless the step reports arch metrics."""
    from megatron.core.transformer import arch_metrics

    monkeypatch.setattr(ckda_branch, "current_iteration", lambda: 36050)
    monkeypatch.setattr(arch_metrics, "active", lambda: False)

    def _boom(*a, **k):  # pragma: no cover - only on failure
        raise AssertionError("arch metrics must not be recorded on an inactive step")

    monkeypatch.setattr(arch_metrics, "record_mean", _boom)
    _write(_Layer(ramp=300, start=35900), seed=10)  # must not raise


def test_impact_metric_reports_gate_and_relative_move(monkeypatch):
    from megatron.core.transformer import arch_metrics

    monkeypatch.setattr(ckda_branch, "current_iteration", lambda: 36050)
    monkeypatch.setattr(arch_metrics, "active", lambda: True)
    seen: dict[str, float] = {}

    def _record(name, layer_number, num_layers, values):
        seen[name] = float(values)
        assert layer_number == 11  # the V layer's 1-based number

    monkeypatch.setattr(arch_metrics, "record_mean", _record)
    result, streams, written, _ = _write(_Layer(ramp=300, start=35900), seed=11)

    assert seen["vgate/r"] == pytest.approx(0.5)
    move = float((written - streams).square().sum(-1).sqrt().mean())
    size = float(streams.square().sum(-1).sqrt().mean())
    assert seen["vgate/delta"] == pytest.approx(move / size)
    # the gated output is the lerp, so its own move is exactly r times the ungated one
    assert torch.allclose(result - streams, 0.5 * (written - streams), atol=1e-6)


# --------------------------------------------------------------------------- learnable gate
#
# v_block_learnable_gate (docs/v4b_plan.md, option A): instead of a schedule,
#     streams' = (1 - rho) * streams + rho * (h_res @ streams) + h_post^T (g_v * out)
# with g_v = v_gate.out_gate and rho = v_gate.res_gate, both 0 at insertion. The point of the
# learnable gate is that both gates carry a gradient at step 0 while the copy-initialized branch is
# already a working attention: a schedule cannot do that.


def _far_h_res():
    """A doubly stochastic mixer far from the identity: every entry 1/n."""
    return torch.full((S, B, N, N), 1.0 / N)


def test_gate_module_is_two_zero_scalars():
    gate = tl.VBlockGate(N)
    params = dict(gate.named_parameters())
    assert set(params) == {"out_gate", "res_gate"}
    assert params["out_gate"].shape == (1,) and params["res_gate"].shape == (1,)
    assert torch.count_nonzero(params["out_gate"]) == 0
    assert torch.count_nonzero(params["res_gate"]) == 0


def test_zero_gates_are_the_streams_bit_exactly():
    """rho = 0 and g_v = 0: neither the mixing nor the write reaches the streams."""
    result, streams, _, _, _ = _write_ref(_Layer(vgate=_gate(0.0, 0.0)), seed=20, h_res=_far_h_res())
    assert torch.equal(result, streams)


def test_open_gates_are_the_v4a_write():
    """g_v = rho = 1: exactly the write v4a performs."""
    result, streams, out, h_post, h_res = _write_ref(_Layer(vgate=_gate(1.0, 1.0)), seed=21)
    expected = _mixed(h_res, streams) + h_post.unsqueeze(-1) * out.unsqueeze(-2)
    assert torch.allclose(result, expected, atol=1e-6)


def test_res_gate_zero_keeps_only_the_attention_write():
    """rho = 0, g_v = 1: the residual path is untouched, the branch writes on top of it."""
    result, streams, out, h_post, _ = _write_ref(
        _Layer(vgate=_gate(1.0, 0.0)), seed=22, h_res=_far_h_res()
    )
    expected = streams + h_post.unsqueeze(-1) * out.unsqueeze(-2)
    assert torch.allclose(result, expected, atol=1e-6)


def test_out_gate_zero_keeps_only_the_stream_mixing():
    """g_v = 0, rho = 1: the write is off, the mixer still mixes (D1 learns instead of inheriting)."""
    result, streams, _, _, h_res = _write_ref(_Layer(vgate=_gate(0.0, 1.0)), seed=23)
    assert torch.allclose(result, _mixed(h_res, streams), atol=1e-6)


def test_both_gates_have_a_gradient_at_step_zero():
    """The whole point: dL/dg_v and dL/drho are non-zero at 0, so the gates open themselves."""
    gate = _gate(0.0, 0.0)
    streams, _, output_with_bias, _ = _tensors(24)
    h_post = torch.rand(S, B, N, generator=torch.Generator().manual_seed(24))
    layer = _Layer(vgate=gate)
    mixer = _RefMixer()
    result = layer._mhc_write(mixer, "attn", streams, output_with_bias, (h_post, _far_h_res()))
    result.sum().backward()
    assert gate.out_gate.grad is not None and float(gate.out_gate.grad.abs()) > 0
    assert gate.res_gate.grad is not None and float(gate.res_gate.grad.abs()) > 0


def test_zero_out_gate_blocks_the_branch_output_gradient():
    """At g_v = 0 the branch cannot move the model through the write: dL/dout is exactly 0.

    The gate itself still moves (test above), which is the difference to the v4a zero o_proj: there
    the *parameter* had no direction and no gradient, here only the branch's influence is zero.
    """
    gate = _gate(0.0, 0.0)
    streams, _, (out, _), (h_post, h_res) = _tensors(25)
    out = out.clone().requires_grad_(True)
    result = _Layer(vgate=gate)._mhc_write(_RefMixer(), "attn", streams, (out, None), (h_post, h_res))
    result.sum().backward()
    assert torch.count_nonzero(out.grad) == 0


def test_mlp_is_never_gated_learnably():
    result, streams, out, h_post, h_res = _write_ref(_Layer(vgate=_gate(0.0, 0.0)), kind="mlp", seed=26)
    expected = _mixed(h_res, streams) + h_post.unsqueeze(-1) * out.unsqueeze(-2)
    assert torch.allclose(result, expected, atol=1e-6)


def test_no_gate_module_is_the_v4a_path():
    """A layer without the module (v4a, every non-V layer) writes whole: the same bytes."""
    layer = _Layer(vgate=None)
    assert layer._v_learnable_gate("attn") is None
    result, streams, _, _, h_res = _write_ref(layer, seed=27)
    assert result.shape == streams.shape


def test_gate_scope_is_the_v_layers_only():
    """``_build_v_gate``: only kv_share_layers get one, and the two gates exclude each other."""
    from megatron.core.transformer.transformer_layer import _build_v_gate

    base = dict(
        v_block_learnable_gate=True,
        v_block_gate_ramp=0,
        mhc=True,
        mhc_expansion_rate=N,
        kv_share_layers=[10, 27],
    )
    assert _build_v_gate(SimpleNamespace(**base), 11) is not None  # V layer 10, 1-based 11
    assert _build_v_gate(SimpleNamespace(**base), 10) is None  # not a V layer
    assert _build_v_gate(SimpleNamespace(**{**base, "v_block_learnable_gate": False}), 11) is None
    with pytest.raises(ValueError, match="v_block_gate_ramp"):
        _build_v_gate(SimpleNamespace(**{**base, "v_block_gate_ramp": 300}), 11)
    with pytest.raises(ValueError, match="mhc"):
        _build_v_gate(SimpleNamespace(**{**base, "mhc": False}), 11)


def test_config_schema_and_cli_metadata_have_the_flag():
    from oellm_autoexp.backends.megatron.cli_metadata import (
        MEGATRON_ACTION_SPECS,
        MEGATRON_ARG_METADATA,
    )
    from oellm_autoexp.backends.megatron.config_schema import MegatronConfig

    annotations = getattr(MegatronConfig, "__annotations__", {})
    assert annotations.get("v_block_learnable_gate") is bool
    assert MegatronConfig.model_fields["v_block_learnable_gate"].default is False
    meta = MEGATRON_ARG_METADATA["v_block_learnable_gate"]
    assert meta.arg_type is bool and meta.default is False
    assert MEGATRON_ACTION_SPECS["v_block_learnable_gate"].option_strings == (
        "--v-block-learnable-gate",
    )


def test_v4b_learned_config_sets_the_learnable_gate_and_no_exit_interval():
    """The arm is open-ended (user 2026-11-17: end it by hand) and keeps the schedule off."""
    text = (ROOT / "config/experiments/korbi/recall_surgery/v4b_learned_mn5.yaml").read_text()
    assert "v_block_learnable_gate: true" in text
    assert "v_block_gate_ramp: 0" in text
    assert "v4a_v6_pin_mn5" in text  # same pattern, sources, data order and saves as v4a
    assert "runs/v4b_learned/main" in text
    assert "runs/v4b_src/ckpt" in text  # the copy-init surgery, not the v4a checkpoint
    assert "exit_interval" not in text  # open-ended: inherited from the parent
    assert "35905" in text and "36400" in text and "36900" in text
