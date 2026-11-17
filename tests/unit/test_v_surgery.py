"""Checks for the V-layer checkpoint surgery and its parameter accounting.

The real 1.1 TB checkpoint is checked by scripts/korbi/production/verify_v_surgery.py.
These tests use tiny DCP checkpoints to catch the metadata rename-chain bug and to pin the
V-layer tensor shapes / parameter count.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "korbi" / "production"))

import dcp_surgery  # noqa: E402
import insert_v_layers  # noqa: E402
import add_v_mhc  # noqa: E402


def test_pattern_and_v_sources():
    pattern, v_layers = insert_v_layers.build_pattern()
    assert len(pattern) == 104
    assert v_layers == [10, 27, 44, 62, 79, 96]
    assert pattern.replace("V", "") == insert_v_layers.ORIG_PATTERN
    assert pattern.count("V") == 6


def test_v_layer_parameter_count_is_pinned():
    v_all = insert_v_layers.v_tensors(10, 1234)
    v_model = {k: v for k, v in v_all.items() if k.startswith("model.module.")}
    mhc_model, _ = add_v_mhc.mixer_tensors(10)
    per_v = sum(t.numel() for t in v_model.values())
    mhc = sum(t.numel() for t in mhc_model.values())
    assert per_v == 25_168_128, per_v  # qkv + proj + q norm + input norm
    assert mhc == 196_635, mhc  # phi + b_pre/b_post/b_res + 3 alpha scalars
    assert per_v + mhc == 25_364_763
    assert 6 * (per_v + mhc) == 152_188_578


def test_v_and_mhc_shapes_and_identity_init():
    v_all = insert_v_layers.v_tensors(10, 1234)
    model = {k: v for k, v in v_all.items() if k.startswith("model.module.")}
    opt = {k: v for k, v in v_all.items() if k.startswith("optimizer.")}
    assert model["model.module.decoder.layers.10.self_attention.linear_qkv.weight"].shape == (8192, 2048)
    assert model["model.module.decoder.layers.10.self_attention.linear_proj.weight"].shape == (2048, 4096)
    assert torch.count_nonzero(model["model.module.decoder.layers.10.self_attention.linear_proj.weight"]) == 0
    assert torch.all(model["model.module.decoder.layers.10.self_attention.q_layernorm.weight"] == 1)
    assert torch.all(model["model.module.decoder.layers.10.input_layernorm.weight"] == 1)

    mhc_model, mhc_opt = add_v_mhc.mixer_tensors(10)
    assert mhc_model["model.module.decoder.layers.10.mhc_attn_mixer.phi"].shape == (8192, 24)
    assert torch.count_nonzero(mhc_model["model.module.decoder.layers.10.mhc_attn_mixer.phi"]) == 0
    assert torch.allclose(
        mhc_model["model.module.decoder.layers.10.mhc_attn_mixer.b_pre"],
        torch.full((4,), -1.0986122886681098),
    )
    assert torch.all(mhc_model["model.module.decoder.layers.10.mhc_attn_mixer.b_res"] == 5 * torch.eye(4))
    assert len(model) + len(mhc_model) == 11
    assert len(opt) + len(mhc_opt) == 31
    assert "optimizer.0.state.module.module.module.decoder.layers.10.self_attention.linear_proj.weight.g" in opt
    assert "optimizer.0.state.module.module.module.decoder.layers.10.input_layernorm.weight.exp_avg" in opt
    assert not any("self_attention.self_attention" in k for k in opt)


def test_rename_chain_preserves_every_key_and_value(tmp_path):
    """A shift 26->28, 28->30, 30->32 must not lose source 28/30 metadata (the V1 bug)."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    keys = [
        "model.module.decoder.layers.26.pre_mlp_layernorm.weight",
        "model.module.decoder.layers.28.pre_mlp_layernorm.weight",
        "model.module.decoder.layers.30.pre_mlp_layernorm.weight",
    ]
    values = {
        keys[0]: torch.full((4,), 1.0),
        keys[1]: torch.full((4,), 2.0),
        keys[2]: torch.full((4,), 3.0),
    }
    dcp.save(values, storage_writer=FileSystemWriter(src), no_dist=True)
    md = FileSystemReader(src).read_metadata()
    rename = {keys[0]: "model.module.decoder.layers.28.pre_mlp_layernorm.weight",
              keys[1]: "model.module.decoder.layers.30.pre_mlp_layernorm.weight",
              keys[2]: "model.module.decoder.layers.32.pre_mlp_layernorm.weight"}
    out = dcp_surgery.write_surgery(src, dst, md, {}, tag="chain", rename=rename, thread_count=1)

    want = {rename[k]: v for k, v in values.items()}
    assert set(out.state_dict_metadata) == set(want)
    loaded = {k: torch.empty_like(v) for k, v in want.items()}
    dcp.load(loaded, storage_reader=FileSystemReader(dst))
    for k, v in want.items():
        assert torch.equal(loaded[k], v), k


@pytest.mark.skipif(not __import__("os").environ.get("V_SURGERY_SRC"), reason="real checkpoint not configured")
def test_real_checkpoint_conversion():
    import subprocess
    src = __import__("os").environ["V_SURGERY_SRC"]
    dst = __import__("os").environ["V_SURGERY_DST"]
    subprocess.run([sys.executable, str(ROOT / "scripts/korbi/production/verify_v_surgery.py"),
                    "--src", src, "--dst", dst, "--sample-values"], check=True)
