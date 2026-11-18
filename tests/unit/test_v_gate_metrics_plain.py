"""The V-gate arch metrics must read plain, aligned tensors and must never take an arm down.

The 36000 crash of v4b: ``vgate.out_gate`` is a one-element FSDP2 ``DTensor``. Its local shard is
empty on all but one data-parallel rank, and a shard view into the flat parameter buffer faults
the vectorized metric kernels with ``CUDA error: misaligned address``. The fix records through
``_plain_gate`` (``to_local`` + ``contiguous``) and wraps the whole recording so a diagnostic
can never kill a run. Gathering instead is a second, independent fault source: a collective
issued from inside the forward while the parameter all-gather is in flight.

These tests pin the three pieces that make that true:

* ``_plain_gate`` / ``_plain_weight`` return plain contiguous tensors from DTensor-like inputs;
* ``_v_param_row_norms`` reports the branch row norms (q and gate apart, o whole);
* ``_record_v_gate`` swallows a recording failure, latches it once, and records nothing after.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
tl = pytest.importorskip("megatron.core.transformer.transformer_layer")
arch_metrics = pytest.importorskip("megatron.core.transformer.arch_metrics")


class _DTensorLike:
    """The two DTensor methods the metric path uses, with the shard being an empty slice."""

    def __init__(self, full: torch.Tensor, local: torch.Tensor) -> None:
        self._full = full
        self._local = local
        self.gathered = 0
        self.localised = 0

    def detach(self) -> "_DTensorLike":
        return self

    def float(self) -> "_DTensorLike":
        return self

    def full_tensor(self) -> torch.Tensor:
        self.gathered += 1
        return self._full

    def to_local(self) -> torch.Tensor:
        self.localised += 1
        return self._local


def test_plain_gate_reads_the_local_shard_and_never_gathers():
    """A gather from inside the forward is the fault: an empty local shard is the correct input."""
    param = _DTensorLike(torch.tensor([0.25]), local=torch.empty(0))
    out = tl._plain_gate(param)
    assert param.localised == 1
    assert param.gathered == 0, "full_tensor is a collective inside the forward and faults"
    assert out.is_contiguous()
    assert out.numel() == 0, "this rank owns no element; collect reconstructs the value"


def test_plain_gate_returns_the_value_on_the_rank_that_owns_it():
    param = _DTensorLike(torch.tensor([0.25]), local=torch.tensor([0.25]))
    out = tl._plain_gate(param)
    assert param.gathered == 0
    assert out.is_contiguous() and out.numel() == 1
    assert out.item() == pytest.approx(0.25)


def test_plain_gate_accepts_a_plain_parameter_and_makes_it_contiguous():
    plain = torch.arange(4.0).reshape(2, 2).t()  # non-contiguous view
    assert not plain.is_contiguous()
    out = tl._plain_gate(plain)
    assert out.is_contiguous()
    assert torch.equal(out, plain)


def test_plain_weight_keeps_the_local_rows_of_a_dtensor():
    local = torch.ones(3, 4)
    weight = _DTensorLike(torch.ones(12, 4), local=local)
    out = tl._plain_weight(weight)
    assert weight.localised == 1 and weight.gathered == 0, "a sharded weight must not be gathered"
    assert out.shape == (3, 4) and out.is_contiguous()


def test_v_param_row_norms_splits_q_and_gate_per_query_group():
    groups, per, in_dim = 2, 3, 5
    out_dim = groups * 2 * per
    w = torch.zeros(out_dim, in_dim)
    v = w.view(groups, 2, per, in_dim)
    # every q row of group 0 has norm 2, of group 1 norm 4; gate rows 3 and 5
    v[0, 0, :, 0] = 2.0
    v[0, 1, :, 0] = 3.0
    v[1, 0, :, 0] = 4.0
    v[1, 1, :, 0] = 5.0
    norms = tl._v_param_row_norms(w, groups, out_dim)
    assert set(norms) == {"q", "gate"}
    assert norms["q"].item() == pytest.approx((2.0 + 4.0) / 2)
    assert norms["gate"].item() == pytest.approx((3.0 + 5.0) / 2)


def test_v_param_row_norms_reports_one_number_without_groups():
    w = torch.zeros(4, 3)
    w[0] = torch.tensor([1.0, 0.0, 0.0])
    norms = tl._v_param_row_norms(w)
    assert set(norms) == {"qkv"}
    assert norms["qkv"].item() == pytest.approx(1.0 / 4)


def _layer():
    return SimpleNamespace(
        layer_number=7,
        config=SimpleNamespace(num_layers=8),
        self_attention=None,
    )


def _vgate():
    return SimpleNamespace(out_gate=torch.tensor([0.1]), res_gate=torch.tensor([0.2]))


S, B, N, C = 2, 3, 4, 8


def test_record_v_gate_does_nothing_when_the_step_reports_no_metrics(monkeypatch):
    calls = []
    monkeypatch.setattr(arch_metrics, "active", lambda: False)
    monkeypatch.setattr(arch_metrics, "record_mean", lambda *a, **k: calls.append(a))
    streams, written = torch.zeros(S, B, N, C), torch.ones(S, B, N, C)
    tl.TransformerLayer._record_v_gate(_layer(), _vgate(), streams, written)
    assert calls == []


def test_record_v_gate_swallows_a_recording_error_and_latches(monkeypatch):
    """A metric that raises must not reach the training step, and must switch itself off once."""
    monkeypatch.setattr(tl, "_VGATE_METRIC_ERROR", None)
    monkeypatch.setattr(arch_metrics, "active", lambda: True)
    monkeypatch.setattr(arch_metrics, "num_layers_of", lambda config: 8)
    calls = []

    def boom(*a, **k):
        calls.append(a)
        raise RuntimeError("CUDA error: misaligned address")

    monkeypatch.setattr(arch_metrics, "record_mean", boom)
    streams, written = torch.zeros(S, B, N, C), torch.ones(S, B, N, C)
    tl.TransformerLayer._record_v_gate(_layer(), _vgate(), streams, written)
    assert calls, "the failing metric must have been attempted"
    assert tl._VGATE_METRIC_ERROR is not None
    assert "misaligned address" in tl._VGATE_METRIC_ERROR

    # latched: a second step records nothing at all
    calls.clear()
    tl.TransformerLayer._record_v_gate(_layer(), _vgate(), streams, written)
    assert calls == []


def test_record_v_gate_records_plain_values(monkeypatch):
    monkeypatch.setattr(tl, "_VGATE_METRIC_ERROR", None)
    monkeypatch.setattr(arch_metrics, "active", lambda: True)
    monkeypatch.setattr(arch_metrics, "num_layers_of", lambda config: 8)
    seen = {}
    monkeypatch.setattr(
        arch_metrics, "record_mean", lambda key, layer, n, value: seen.setdefault(key, value)
    )
    streams, written = torch.zeros(S, B, N, C), torch.ones(S, B, N, C)
    tl.TransformerLayer._record_v_gate(_layer(), _vgate(), streams, written)
    assert tl._VGATE_METRIC_ERROR is None
    assert set(seen) == {"vgate/out", "vgate/res", "vgate/delta"}
    assert seen["vgate/out"].item() == pytest.approx(0.1)
    assert seen["vgate/res"].item() == pytest.approx(0.2)
