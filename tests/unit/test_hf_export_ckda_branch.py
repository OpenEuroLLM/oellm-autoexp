"""HF export of a ckda_branch run past its hand-over: the branch layers export as standard attention layers
(the branch's projections, the layer's input norm, m * gamma_scale * gamma folded into o_proj)."""
import pytest
import torch

from oellm_autoexp.hf_export.export_ambitious import _MHC, branch_fold, convert_layer

HID, HEADS, GROUPS, D = 32, 4, 2, 16


def _m(**kw):
    m = dict(ckda_branch="attention", ckda_branch_layers=[1], ckda_branch_num_query_groups=GROUPS,
             ckda_branch_start_iteration=100, ckda_branch_alpha_hold=25, ckda_branch_alpha_ramp=175,
             ckda_branch_alpha_end=0.0, ckda_branch_m_start=4.0, ckda_branch_m_ramp=200,
             ckda_branch_gamma_scale=10.0, num_query_groups=GROUPS, group_query_attention=True,
             num_attention_heads=HEADS, _ckpt_iteration=300)
    m.update(kw)
    return m


def test_branch_fold_rules():
    assert branch_fold({}) == (set(), 1.0)
    assert branch_fold(_m()) == ({1}, 10.0)  # after the m ramp: m = 1
    with pytest.raises(ValueError, match="inside the hand-over"):
        branch_fold(_m(_ckpt_iteration=200))
    with pytest.raises(ValueError, match="per-layer KV groups"):
        branch_fold(_m(ckda_branch_num_query_groups=8))
    with pytest.raises(ValueError, match="only attention"):
        branch_fold(_m(ckda_branch="kronlstm"))
    with pytest.raises(ValueError, match="alpha_end"):
        branch_fold(_m(ckda_branch_alpha_end=0.5))


def test_folded_layer_is_an_attention_layer():
    g = torch.Generator().manual_seed(0)
    p = "decoder.layers.1."
    rows = GROUPS * (2 * HEADS // GROUPS + 2) * D
    t = {p + f"mhc_attn_mixer.{n}": torch.randn(3, generator=g) for n in _MHC}
    t[p + "input_layernorm.weight"] = torch.rand(HID, generator=g)
    t[p + "self_attention.layer.q_proj.weight"] = torch.randn(HID, HID, generator=g)  # complex_kda, weight 0
    t[p + "self_attention.layer.A_log"] = torch.randn(HEADS, generator=g)
    t[p + "self_attention.branch.linear_qkv.weight"] = torch.randn(rows, HID, generator=g)
    t[p + "self_attention.branch.linear_proj.weight"] = torch.randn(HID, HEADS * D, generator=g)
    t[p + "self_attention.branch.q_layernorm.weight"] = torch.rand(D, generator=g)
    t[p + "self_attention.branch.k_layernorm.weight"] = torch.rand(D, generator=g)
    t[p + "self_attention.branch_mix.gamma"] = torch.tensor([0.07])
    src = dict(t)
    cfg = {"block_types": ["attention", "attention"], "residual": "mhc", "num_attention_heads": HEADS,
           "num_key_value_heads": GROUPS, "head_dim": D, "attention_output_gate": True}
    out = {}
    convert_layer(1, t, cfg, _m(), out.__setitem__, torch.float32, torch.float32, "bf16")
    hp = "model.layers.1."
    assert not t  # every tensor mapped or dropped
    assert torch.equal(out[hp + "input_layernorm.weight"], src[p + "input_layernorm.weight"])
    assert torch.allclose(out[hp + "mixer.o_proj.weight"], src[p + "self_attention.branch.linear_proj.weight"] * 0.7)
    assert torch.equal(out[hp + "mixer.q_norm.weight"], src[p + "self_attention.branch.q_layernorm.weight"])
    assert out[hp + "mixer.q_proj.weight"].shape == (HEADS * D, HID)
    assert out[hp + "mixer.k_proj.weight"].shape == (GROUPS * D, HID)
    assert not any("layer." in k for k in out if k.startswith(hp + "mixer."))
