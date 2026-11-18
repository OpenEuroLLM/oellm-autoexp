"""HF export of a first-class V hybrid layer (pattern symbol V, docs/recall_surgery_ambitious.md §22).

A V layer is an attention TransformerLayer with a separate input norm and only q / gate in its
``linear_qkv`` (the source attention layer's keys / values are shared at run time). The export
must write q / gate / o / q-norm on the layer's normalized input and no k / v projections.
"""
import pytest
import torch

from oellm_autoexp.hf_export.export_ambitious import _MHC, _split_kvshare, convert_layer, kv_share_input_norm

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


def test_v_layer_without_input_norm_exports_no_norm():
    """A checkpoint of the pre-§22.18 V layer spec has no input norm: the export adds none either."""
    t, out = _tensors(), {}
    del t[f"decoder.layers.{LAYER}.input_layernorm.weight"]
    convert_layer(LAYER, t, _cfg() | {"kv_share_input_norm": False}, {},
                  lambda name, w: out.__setitem__(name, w), torch.float32, torch.float32, "bf16")
    hp = f"model.layers.{LAYER}."
    assert hp + "input_layernorm.weight" not in out
    assert sorted(out) == sorted(
        [hp + "mixer.q_proj.weight", hp + "mixer.gate_proj.weight", hp + "mixer.o_proj.weight",
         hp + "mixer.q_norm.weight"]
        + [hp + f"mhc.{n}" for n in _MHC]
    )
    assert t == {}  # every training tensor of the layer is consumed


def test_kv_share_input_norm_reads_the_checkpoint():
    cfg = _cfg()
    norm = {f"decoder.layers.{LAYER}.input_layernorm.weight": None}
    assert kv_share_input_norm(norm, cfg) is True
    assert kv_share_input_norm({}, cfg) is False
    # two V layers, the norm on one of them: no V layer spec produced that
    with pytest.raises(ValueError, match="mixed checkpoint"):
        kv_share_input_norm(norm, dict(cfg, block_types=cfg["block_types"] + ["kvshare"]))


# ------------------------------------------------------------------- v_block_learnable_gate
#
# An inserted V layer trained with the two learnable gates carries v_gate.out_gate (g_v) and
# v_gate.res_gate (rho). g_v folds into o_proj exactly as a ckda_branch's gamma does; rho cannot
# fold into the mixer's logits (it pulls h_res toward the identity after the sinkhorn), so it is
# exported in config.json as kv_share_res_gate and the model applies it to h_res.


def _gated_tensors(out_gate: float = 0.5, res_gate: float = 0.25):
    t = _tensors()
    t[f"decoder.layers.{LAYER}.v_gate.out_gate"] = torch.tensor([out_gate])
    t[f"decoder.layers.{LAYER}.v_gate.res_gate"] = torch.tensor([res_gate])
    return t


def test_v_layer_folds_g_v_into_o_proj_and_exports_rho():
    t = _gated_tensors(out_gate=0.5, res_gate=0.25)
    o_before = t[f"decoder.layers.{LAYER}.self_attention.linear_proj.weight"].float().clone()
    o_before[0, 0] = 2.0  # not the zero projection of v4a: g_v must scale real weights
    t[f"decoder.layers.{LAYER}.self_attention.linear_proj.weight"] = o_before
    cfg, out = _cfg(), {}
    convert_layer(LAYER, t, cfg, {}, lambda name, w: out.__setitem__(name, w),
                  torch.float32, torch.float32, "bf16")
    hp = f"model.layers.{LAYER}."
    assert torch.allclose(out[hp + "mixer.o_proj.weight"], 0.5 * o_before, atol=1e-6)
    assert cfg["kv_share_res_gate"] == {str(LAYER): 0.25}
    assert t == {}  # both gate tensors are consumed (g_v folded, rho in the config)


def test_v_layer_export_warns_on_a_nearly_closed_gate(capsys):
    """A branch whose g_v is still ~0 exports an almost-v4 model: warn, do not refuse.

    The number is valid for the model as it is (the gate is folded exactly); it is only a readout
    warning. The gate value, not a fixed iteration, decides -- the gates open at Adam's scalar step.
    """
    for gate in (0.0, 0.01, -0.03):
        out = {}
        convert_layer(LAYER, _gated_tensors(out_gate=gate), _cfg(), {"_ckpt_iteration": 36400},
                      lambda name, w: out.__setitem__(name, w), torch.float32, torch.float32, "bf16")
        assert f"model.layers.{LAYER}.mixer.o_proj.weight" in out
    assert capsys.readouterr().err.count("almost-v4 readout") == 3


def test_v_layer_export_allows_an_open_gate():
    for gate in (0.5, -0.5, 1.0):
        out = {}
        convert_layer(LAYER, _gated_tensors(out_gate=gate), _cfg(), {"_ckpt_iteration": 36400},
                      lambda name, w: out.__setitem__(name, w), torch.float32, torch.float32, "bf16")
        assert f"model.layers.{LAYER}.mixer.o_proj.weight" in out


def test_v_layer_without_gates_keeps_the_config_empty_and_the_weight_whole():
    """A v4a checkpoint has no gates: no fold, no rho, and the guard never fires."""
    cfg, out = _cfg(), {}
    convert_layer(LAYER, _tensors(), cfg, {"_ckpt_iteration": 35900},
                  lambda name, w: out.__setitem__(name, w), torch.float32, torch.float32, "bf16")
    assert cfg.get("kv_share_res_gate", {}) == {}
    assert "kv_share_res_gate" not in _cfg()  # the export config starts without it


def test_config_class_accepts_the_res_gate_dict():
    from oellm_autoexp.hf_export.oellm_ambitious.configuration_oellm_ambitious import (
        OellmAmbitiousConfig,
    )

    cfg = OellmAmbitiousConfig(kv_share_res_gate={10: 0.25})
    assert cfg.kv_share_res_gate == {"10": 0.25}
    assert OellmAmbitiousConfig().kv_share_res_gate == {}


def test_schedule_ramp_window_is_refused_from_the_run_config():
    """A checkpoint inside the schedule ramp exports a different model than the arm trained.

    The window comes from the run's own `v_block_gate_ramp` / `v_block_gate_start_iteration`, so a
    100-step ramp (v4c) is refused at 36050 and allowed at 36000+100, and a 300-step one (v4b_gate)
    is refused at 36100.
    """
    import pytest

    t = _tensors() | {f"decoder.layers.{LAYER}.v_gate.res_gate": torch.zeros(1)}
    cfg = _cfg() | {"kv_share_input_norm": True}

    def export(iteration, ramp, start=35900):
        m = {"_ckpt_iteration": iteration, "v_block_gate_ramp": ramp,
             "v_block_gate_start_iteration": start}
        convert_layer(LAYER, dict(t), cfg, m, lambda name, w: None,
                      torch.float32, torch.float32, "bf16")

    with pytest.raises(ValueError, match="schedule ramp runs"):
        export(35950, 100)          # inside a 100-step ramp [35900, 36000)
    export(36000, 100)              # at the ramp's end: allowed
    with pytest.raises(ValueError, match="schedule ramp runs"):
        export(36100, 300)          # inside a 300-step ramp [35900, 36200)
    export(35950, 0)                # no ramp: allowed
