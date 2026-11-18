"""The HF/vLLM port of v4b's learnable residual gate `rho` (docs/v4b_plan.md §9).

`g_v` needs no model support: the export folds it into `o_proj`. `rho` cannot fold -- it pulls the
mHC stream mixer toward the identity *after* the sinkhorn -- so it travels in `config.json` as
`kv_share_res_gate` and the model applies `h_res = lerp(I, h_res, rho)`. This test pins the HF side;
the vLLM side has the same three lines (`_gated`, applied in both `_mhc_post` and `_mhc_forward`).
"""
import sys

import pytest
import torch

from oellm_autoexp.hf_export.oellm_hybrid import configuration_oellm_hybrid as _ch
from oellm_autoexp.hf_export.oellm_hybrid import modeling_oellm_hybrid as _mh

sys.modules.setdefault("oellm_autoexp.hf_export.oellm_ambitious.configuration_oellm_hybrid", _ch)
sys.modules.setdefault("oellm_autoexp.hf_export.oellm_ambitious.modeling_oellm_hybrid", _mh)

from oellm_autoexp.hf_export.oellm_ambitious.configuration_oellm_ambitious import (  # noqa: E402
    OellmAmbitiousConfig,
)
from oellm_autoexp.hf_export.oellm_ambitious.modeling_oellm_ambitious import (  # noqa: E402
    OellmAmbitiousLayer,
)

HID, LAYER = 32, 1


def _layer(rho):
    torch.manual_seed(0)
    cfg = OellmAmbitiousConfig(
        vocab_size=64, hidden_size=HID, num_hidden_layers=2, num_attention_heads=4, head_dim=8,
        num_key_value_heads=2, block_types=["attention", "kvshare"], kv_share_sources={str(LAYER): 0},
        residual="mhc", mhc_expansion_rate=4, rms_norm_eps=1e-5,
        kv_share_res_gate=None if rho is None else {str(LAYER): rho},
    )
    layer = OellmAmbitiousLayer(cfg, LAYER)
    layer._mix = lambda *a, **k: (torch.zeros(2, 5, HID), None)  # the gate is what is under test
    return layer


def _pending(layer, streams):
    """The (h_post, h_res) the layer hands to the mHC write."""
    seen = {}

    def capture(streams, out, pending):
        seen["pending"] = pending
        return streams

    layer.mhc.write = capture
    layer.forward(streams)
    return seen["pending"]


def test_kvshare_layer_pulls_h_res_toward_the_identity():
    streams = torch.randn(2, 5, 4, HID)
    h_post0, h_res0 = _pending(_layer(None), streams)
    h_post1, h_res1 = _pending(_layer(0.5), streams)
    assert torch.allclose(h_post1, h_post0)  # rho touches the mixer only, not the write coefficient
    eye = torch.eye(4, dtype=h_res0.dtype)
    assert torch.allclose(h_res1, torch.lerp(eye, h_res0, 0.5), atol=1e-6)
    # the un-gated layer is the parent's mixing exactly, the gated one moves toward I
    assert (h_res1 - eye).abs().max() < (h_res0 - eye).abs().max()


def test_other_layers_and_missing_gates_are_untouched():
    streams = torch.randn(2, 5, 4, HID)
    # no kv_share_res_gate entry at all: a checkpoint without the learnable gates (v4a, v4c)
    torch.manual_seed(0)
    cfg = OellmAmbitiousConfig(
        vocab_size=64, hidden_size=HID, num_hidden_layers=2, num_attention_heads=4, head_dim=8,
        num_key_value_heads=2, block_types=["attention", "kvshare"], kv_share_sources={str(LAYER): 0},
        residual="mhc", mhc_expansion_rate=4, rms_norm_eps=1e-5,
    )
    layer = OellmAmbitiousLayer(cfg, LAYER)
    assert layer.res_gate is None
    layer._mix = lambda *a, **k: (torch.zeros(2, 5, HID), None)
    _, h_res = _pending(layer, streams)
    assert h_res is not None and h_res.shape[-2:] == (4, 4)
