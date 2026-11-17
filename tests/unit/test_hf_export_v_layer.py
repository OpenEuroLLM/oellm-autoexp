"""HF export of a first-class V hybrid layer (pattern symbol V, docs/recall_surgery_ambitious.md §22).

A V layer is an attention TransformerLayer with a separate input norm and only q / gate in its
``linear_qkv`` (the source attention layer's keys / values are shared at run time). The export
must write q / gate / o / q-norm on the layer's normalized input and no k / v projections.
"""
import torch

from oellm_autoexp.hf_export.export_ambitious import _MHC, _split_kvshare, convert_layer

HID, HEADS, GROUPS, D = 16, 4, 2, 8
LAYER = 10


def _cfg():
    types = ["attention"] * LAYER + ["kvshare"]
    return dict(block_types=types, num_attention_heads=HEADS, num_key_value_heads=GROUPS,
                head_dim=D, attention_output_gate=True, residual="mhc", kv_share_sources={"10": 0})


def _tensors():
    p = f"decoder.layers.{LAYER}."
    rows = GROUPS * 2 * (HEADS // GROUPS) * D  # [q | gate] per query group
    qkv = torch.arange(rows * HID, dtype=torch.float32).reshape(rows, HID)
    t = {p + "self_attention.linear_qkv.weight": qkv,
         p + "self_attention.linear_proj.weight": torch.zeros(HID, HEADS * D),
         p + "self_attention.q_layernorm.weight": torch.ones(D),
         p + "input_layernorm.weight": torch.ones(HID)}
    for n in _MHC:
        t[p + f"mhc_attn_mixer.{n}"] = torch.randn(3)
    return t


def test_v_layer_exports_q_gate_o_and_no_kv():
    t, out = _tensors(), {}
    convert_layer(LAYER, t, _cfg(), {}, lambda name, w: out.__setitem__(name, w),
                  torch.float32, torch.float32, "bf16")
    hp = f"model.layers.{LAYER}."
    assert sorted(out) == sorted(
        [hp + "mixer.q_proj.weight", hp + "mixer.gate_proj.weight", hp + "mixer.o_proj.weight",
         hp + "mixer.q_norm.weight", hp + "input_layernorm.weight"]
        + [hp + f"mhc.{n}" for n in _MHC])
    assert not any("k_proj" in k or "v_proj" in k or "k_norm" in k for k in out)
    assert t == {}  # every training tensor of the layer is consumed


def test_v_layer_q_and_gate_come_from_the_per_group_qkv_layout():
    t = _tensors()
    qkv = t[f"decoder.layers.{LAYER}.self_attention.linear_qkv.weight"]
    q, z = _split_kvshare(qkv, _cfg())
    hpg = HEADS // GROUPS
    rows = qkv.view(GROUPS * 2 * hpg, D, -1)
    want_q = torch.cat([torch.arange(2 * hpg * gi, 2 * hpg * gi + hpg) for gi in range(GROUPS)])
    assert torch.equal(q, rows[want_q].reshape(HEADS * D, -1))
    assert torch.equal(z, rows[want_q + hpg].reshape(HEADS * D, -1))
    # the two halves are disjoint and cover the whole [q | gate] block
    assert q.shape == z.shape == (HEADS * D, HID)


def test_v_layer_without_input_norm_fails_loudly():
    """Checkpoints of a V layer built without the input norm cannot be exported faithfully."""
    import pytest

    t = _tensors()
    del t[f"decoder.layers.{LAYER}.input_layernorm.weight"]
    with pytest.raises(ValueError, match="without input_layernorm.weight"):
        convert_layer(LAYER, t, _cfg(), {}, lambda name, w: None,
                      torch.float32, torch.float32, "bf16")
