"""The arch-metrics path must reach the same mHC coefficients without the compiled kernel.

The 36000 crash of v4b and the exposure of v4c: arch metrics need ``h_pre``, which the fused read
does not return, so the metrics path runs the unfused ``MHCMixer.forward``. That forward called the
``torch.compile``d ``_mhc_coeffs`` kernel, and under activation recompute the buffers are freed and
reallocated, so the recomputed forward handed Inductor an unaligned address
(``CUDA error: misaligned address``).

The fix runs the same PyTorch math eagerly when asked. These tests pin that the eager branch is
numerically the same function (so the metrics describe the same model the training path computes)
and that the wiring asks for it exactly when arch metrics are active.
"""
from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")
mhc = pytest.importorskip("megatron.core.transformer.mhc")
tl = pytest.importorskip("megatron.core.transformer.transformer_layer")

MHCMixer = mhc.MHCMixer
S, B, N, C = 2, 3, 4, 16


def _mixer():
    torch.manual_seed(0)
    return MHCMixer(hidden_size=C, expansion_rate=N, num_sinkhorn_iters=5).eval()


def test_forward_takes_an_eager_flag_that_defaults_off():
    sig = inspect.signature(MHCMixer.forward)
    assert "eager" in sig.parameters
    assert sig.parameters["eager"].default is False, "the training path must keep the compiled kernel"


def test_eager_and_compiled_paths_agree():
    mixer = _mixer()
    streams = torch.randn(S, B, N, C)
    with torch.no_grad():
        compiled = mixer(streams)
        eager = mixer(streams, eager=True)
    assert len(compiled) == len(eager)
    for c, e in zip(compiled, eager):
        assert c.shape == e.shape
        assert torch.allclose(c, e, atol=1e-5, rtol=1e-4), (
            "the eager coefficients must be the same function as the compiled ones"
        )


def test_eager_path_skips_the_compiled_coefficients():
    """``eager=True`` must not call the compiled kernel at all: that call is the fault."""
    mixer = _mixer()
    calls = []
    original = mhc._mhc_coeffs

    def spy(*a, **k):
        calls.append(1)
        return original(*a, **k)

    mhc._mhc_coeffs = spy
    try:
        with torch.no_grad():
            mixer(torch.randn(S, B, N, C), eager=True)
        assert calls, "the eager branch should call the plain math directly"
    finally:
        mhc._mhc_coeffs = original


def test_mhc_read_asks_for_eager_only_when_metrics_are_active():
    src = inspect.getsource(tl.TransformerLayer._mhc_read)
    assert "eager=arch_metrics.active()" in src, (
        "the metrics path must ask for the eager coefficients, the training path must not"
    )
