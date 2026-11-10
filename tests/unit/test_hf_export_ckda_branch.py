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
    with pytest.raises(ValueError, match="attention or kronlstm"):
        branch_fold(_m(ckda_branch="mamba"))
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
    # crossfade: the weight is 1 + scale * gamma (scale 10 here: 1.7)
    out = {}
    convert_layer(1, dict(src), cfg, _m(ckda_branch_crossfade=True, _ckpt_iteration=300), out.__setitem__,
                  torch.float32, torch.float32, "bf16")
    assert torch.allclose(out[hp + "mixer.o_proj.weight"], src[p + "self_attention.branch.linear_proj.weight"] * 1.7)
    # calibrated: times the measured scale
    out, cal = {}, dict(src)
    cal[p + "self_attention.branch_mix.calib"] = torch.tensor([2.5])
    convert_layer(1, cal, cfg, _m(ckda_branch_crossfade=True, _ckpt_iteration=300), out.__setitem__,
                  torch.float32, torch.float32, "bf16")
    assert torch.allclose(out[hp + "mixer.o_proj.weight"],
                          src[p + "self_attention.branch.linear_proj.weight"] * 1.7 * 2.5)


def _kron_m(**kw):
    m = _m(ckda_branch="kronlstm", ckda_branch_crossfade=True, ckda_branch_calibrate=True, hidden_size=HID,
           linear_key_head_dim=D, linear_num_key_heads=HEADS, linear_conv_kernel_dim=4, linear_drop_qkv_silu=True,
           linear_fla_allow_neg_eigval=True, linear_fla_gate="signed_sigmoid2", linear_fla_lower_bound=-5.0,
           linear_kronlstm_output_gate="lowrank", layernorm_epsilon=1e-5)
    m.update(kw)
    return m


def test_kronlstm_branch_folds_into_a_kronlstm_block():
    from oellm_autoexp.hf_export.export_ambitious import _kronlstm_layer_spec

    assert branch_fold(_kron_m(ckda_branch_num_query_groups=8)) == ({1}, 10.0)  # KV groups irrelevant here
    g = torch.Generator().manual_seed(1)
    p, hp = "decoder.layers.1.", "model.layers.1."
    t = {p + f"mhc_attn_mixer.{n}": torch.randn(3, generator=g) for n in _MHC}
    t[p + "input_layernorm.weight"] = torch.rand(HID, generator=g)
    t[p + "self_attention.layer.q_proj.weight"] = torch.randn(HID, HID, generator=g)
    for n, shape in (("qkv_proj.weight", (3 * HID, HID)), ("o_proj.weight", (HID, HID)), ("b_proj.weight", (HEADS, HID)),
                     ("c_proj.weight", (HEADS, HID)), ("A_log", (HEADS,))):
        t[p + "self_attention.branch.layer." + n] = torch.randn(*shape, generator=g)
    t[p + "self_attention.branch_mix.gamma"] = torch.tensor([0.1])
    t[p + "self_attention.branch_mix.calib"] = torch.tensor([2.0])
    src = dict(t)
    cfg = {"block_types": ["attention", "kronlstm"], "residual": "mhc"}
    out = {}
    convert_layer(1, t, cfg, _kron_m(), out.__setitem__, torch.float32, torch.float32, "bf16")
    assert not t
    # crossfade, calibrated: weight = calib * (1 + gamma_scale * gamma) = 2 * (1 + 10 * 0.1) = 4
    assert torch.allclose(out[hp + "mixer.layer.o_proj.weight"], src[p + "self_attention.branch.layer.o_proj.weight"] * 4)
    assert torch.equal(out[hp + "mixer.layer.qkv_proj.weight"], src[p + "self_attention.branch.layer.qkv_proj.weight"])
    assert torch.equal(out[hp + "input_layernorm.weight"], src[p + "input_layernorm.weight"])
    assert not any(".layer.q_proj" in k for k in out)  # the weight-0 complex_kda is dropped
    spec = _kronlstm_layer_spec(_kron_m())
    assert spec["class"] == "fla.layers.kronlstm.KronLSTM"
    kw = spec["kwargs"]
    assert kw["drop_silu"] and kw["output_gate_rank"] == "lowrank" and kw["gate"] == "signed_sigmoid2"
    assert kw["allow_neg_eigval"] and kw["decouple_erase"] and not kw["unit_write"] and kw["normalize_v"]
    try:
        import inspect

        from fla.layers.kronlstm import KronLSTM
    except Exception:  # the FLA fork is not on the path
        return
    assert set(kw) <= set(inspect.signature(KronLSTM.__init__).parameters)
