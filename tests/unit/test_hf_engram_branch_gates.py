"""HF port of the per-stream Engram gates (Megatron engram_branch_gates / engram_branch_query / engram_gate_logit_scale /
engram_gate_query_eps; docs/engram_main_run_trial_plan.md).

The gate math is checked against Megatron's own EngramLayer methods; the layer (gates, per-stream conv, per-stream mHC
write) by its function-preserving cases: per-stream copies with the read query equal the single gate, query_mix 0 / 1
equal the read / stream queries, logit_scale 1 changes nothing, and the cached decode equals the full sequence.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "submodules" / "Megatron-LM"))

from oellm_autoexp.hf_export.oellm_hybrid import configuration_oellm_hybrid as _ch  # noqa: E402
from oellm_autoexp.hf_export.oellm_hybrid import modeling_oellm_hybrid as _mh  # noqa: E402

sys.modules.setdefault("oellm_autoexp.hf_export.oellm_ambitious.configuration_oellm_hybrid", _ch)
sys.modules.setdefault("oellm_autoexp.hf_export.oellm_ambitious.modeling_oellm_hybrid", _mh)

from oellm_autoexp.hf_export.export_ambitious import _MHC, convert_layer  # noqa: E402
from oellm_autoexp.hf_export.oellm_ambitious.configuration_oellm_ambitious import (  # noqa: E402
    OellmAmbitiousConfig,
)
from oellm_autoexp.hf_export.oellm_ambitious.modeling_oellm_ambitious import OellmAmbitiousLayer  # noqa: E402

HID, N, VOCAB = 32, 4, 64
ORDERS, HEADS, EMB = [2, 3], 2, 8


def _config(**engram):
    lookups = len(ORDERS) * HEADS
    return OellmAmbitiousConfig(
        vocab_size=VOCAB, hidden_size=HID, num_hidden_layers=1, num_attention_heads=4, head_dim=8,
        num_key_value_heads=2, block_types=["engram"], residual="mhc", mhc_expansion_rate=N, rms_norm_eps=1e-5,
        engram_ngram_orders=ORDERS, engram_num_heads=HEADS, engram_embed_dim=EMB, engram_conv_kernel_size=4,
        engram_table_sizes=[11, 13, 17, 19][:lookups],
        engram_hash_multipliers={"0": [[3 + 2 * j, 5 + 2 * j, 7 + 2 * j][: max(ORDERS)] for j in range(lookups)]},
        engram_gate_signed_sqrt=True, engram_conv_prenorm=True, **engram,
    )


def _layer(**engram):
    torch.manual_seed(0)
    layer = OellmAmbitiousLayer(_config(**engram), 0).float()
    with torch.no_grad():
        for p in layer.parameters():  # non-trivial mHC coefficients, norms and table
            p.copy_(torch.randn_like(p) * (0.3 if p.dim() == 1 else 1.0) + (1.0 if p.dim() == 1 else 0.0))
        layer.mhc.alpha_res.fill_(0.1)
    return layer


def _copy_single_into(branch, single):
    """``branch`` <- ``single``'s weights, every stream's gate a copy of stream 0's."""
    sb, ss = branch.state_dict(), single.state_dict()
    for k in sb:
        if k in ss:
            sb[k] = ss[k].clone()
    m = branch.mixer
    for i in range(m.n_gates - 1):
        sb[f"mixer.key_proj_extra.{i}.weight"] = ss["mixer.key_proj.weight"].clone()
        sb[f"mixer.key_norm_extra.{i}.weight"] = ss["mixer.key_norm.weight"].clone()
        sb[f"mixer.hidden_norm_extra.{i}.weight"] = ss["mixer.hidden_norm.weight"].clone()
    if m.gate_terms is not None and m.gate_terms.query_mix is not None:
        sb["mixer.gate_terms.query_mix"] = torch.zeros(m.n_gates)
    if m.gate_terms is not None and m.gate_terms.logit_scale is not None:
        sb["mixer.gate_terms.logit_scale"] = torch.ones(m.n_gates)
    branch.load_state_dict(sb)


def _inputs(b=2, s=7):
    g = torch.Generator().manual_seed(1)
    return torch.randn(b, s, N, HID, generator=g), torch.randint(0, VOCAB, (b, s), generator=g)


@torch.no_grad()
def test_read_query_copies_equal_the_single_gate():
    single = _layer()
    streams, ids = _inputs()
    want, _ = single(streams, input_ids=ids)
    for query in ("read", "learned"):
        branch = _layer(engram_branch_gates=True, engram_branch_query=query)
        _copy_single_into(branch, single)
        got, _ = branch(streams, input_ids=ids)
        assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), (query, (got - want).abs().max())


@torch.no_grad()
def test_logit_scale_one_changes_nothing():
    single = _layer()
    scaled = _layer(engram_gate_logit_scale=True)
    _copy_single_into(scaled, single)
    streams, ids = _inputs()
    assert torch.allclose(scaled(streams, input_ids=ids)[0], single(streams, input_ids=ids)[0], atol=1e-6)
    with torch.no_grad():
        scaled.mixer.gate_terms.logit_scale.fill_(0.3)
    assert not torch.allclose(scaled(streams, input_ids=ids)[0], single(streams, input_ids=ids)[0], atol=1e-4)


@torch.no_grad()
def test_query_mix_interpolates_read_and_stream():
    streams, ids = _inputs()
    learned = _layer(engram_branch_gates=True, engram_branch_query="learned")
    for query, lam in (("read", 0.0), ("stream", 1.0)):
        ref = _layer(engram_branch_gates=True, engram_branch_query=query)
        ref.load_state_dict({k: v for k, v in learned.state_dict().items() if "query_mix" not in k})
        learned.mixer.gate_terms.query_mix.fill_(lam)
        got, want = learned(streams, input_ids=ids)[0], ref(streams, input_ids=ids)[0]
        assert torch.allclose(got, want, atol=1e-5), (query, (got - want).abs().max())


@torch.no_grad()
def test_streams_get_their_own_outputs():
    branch = _layer(engram_branch_gates=True, engram_branch_query="stream", engram_gate_logit_scale=True)
    streams, ids = _inputs()
    x, pending = branch.mhc.read(streams)
    outs, _ = branch.mixer(x, ids, None, None, streams=streams)
    assert outs.shape == (2, 7, N, HID)
    assert not torch.allclose(outs[..., 0, :], outs[..., 1, :])  # different queries, different gates
    # the per-stream write equals the shared write when all outputs are the same
    same = outs[..., :1, :].expand_as(outs)
    assert torch.allclose(branch.mhc.write_streams(streams, same, pending),
                          branch.mhc.write(streams, outs[..., 0, :], pending), atol=1e-6)


@torch.no_grad()
def test_cached_decode_equals_the_full_sequence():
    branch = _layer(engram_branch_gates=True, engram_branch_query="learned", engram_gate_logit_scale=True,
                    engram_gate_query_eps=1e-12)
    with torch.no_grad():
        branch.mixer.gate_terms.query_mix.copy_(torch.tensor([0.1, -0.2, 0.3, 0.0]))
        branch.mixer.gate_terms.logit_scale.copy_(torch.tensor([0.3, 0.5, 1.2, 0.05]))
    streams, ids = _inputs(s=9)
    full, _ = branch(streams, input_ids=ids)
    a, state = branch(streams[:, :5], input_ids=ids[:, :5])
    b, state = branch(streams[:, 5:8], state=state, input_ids=ids[:, 5:8])
    c, _ = branch(streams[:, 8:], state=state, input_ids=ids[:, 8:])
    assert torch.allclose(torch.cat([a, b, c], dim=1), full, atol=1e-5)


@torch.no_grad()
def test_gate_math_matches_megatron():
    """``_branch_queries`` / ``_branch_gate_logits`` / ``_norm_gated`` against Megatron's EngramLayer methods."""
    from megatron.core.ssm.engram import EngramLayer

    branch = _layer(engram_branch_gates=True, engram_branch_query="learned", engram_gate_logit_scale=True,
                    engram_gate_query_eps=1e-12)
    m = branch.mixer
    m.gate_terms.query_mix.copy_(torch.tensor([0.0, 0.25, -0.5, 1.0]))

    def norm(x):
        return types.SimpleNamespace(weight=x.weight, eps=x.variance_epsilon)

    fake = types.SimpleNamespace(
        branch_query="learned", n_gates=N, gate_terms=types.SimpleNamespace(query_mix=m.gate_terms.query_mix),
        key_proj=m.key_proj, key_proj_extra=list(m.key_proj_extra), hidden_norm=norm(m.hidden_norm),
        key_norm=norm(m.key_norm), hidden_norm_extra=[norm(x) for x in m.hidden_norm_extra],
        key_norm_extra=[norm(x) for x in m.key_norm_extra], conv_norm=norm(m.conv_norm),
    )
    g = torch.Generator().manual_seed(2)
    read, streams = torch.randn(3, 5, HID, generator=g) * 1e-3, torch.randn(3, 5, N, HID, generator=g)
    e = torch.randn(3, 5, m.key_proj.in_features, generator=g)
    q_meg = EngramLayer._branch_queries(fake, read, streams)
    q_hf = m._branch_queries(read, streams)
    assert torch.equal(q_hf, q_meg)
    assert torch.equal(m._branch_gate_logits(q_hf, e), EngramLayer._branch_gate_logits(fake, q_meg, e))
    v, alpha = torch.randn(3, 5, 1, HID, generator=g), torch.rand(3, 5, N, 1, generator=g)
    assert torch.equal(m._norm_gated(v, alpha), EngramLayer._norm_gated(fake, v, alpha))


def test_export_maps_the_per_stream_tensors():
    p, layer = "decoder.layers.0.", 0
    t = {p + n: torch.randn(4, 4) for n in ("key_proj.weight", "value_proj.weight", "hidden_norm.weight",
                                             "key_norm.weight", "conv1d.weight", "conv_norm.weight")}
    for i in range(N - 1):
        for n in ("key_proj_extra", "key_norm_extra", "hidden_norm_extra"):
            t[p + f"{n}.{i}.weight"] = torch.randn(4, 4)
    t[p + "gate_terms.query_mix"] = torch.zeros(N, dtype=torch.bfloat16)
    t[p + "gate_terms.logit_scale"] = torch.full((N,), 0.3, dtype=torch.bfloat16)
    for n in _MHC:
        t[p + f"residual_scheme.mhc_mixer.{n}"] = torch.randn(3)
    cfg = dict(block_types=["engram"], residual="mhc", engram_conv_prenorm=True, engram_sign_hash=False,
               engram_token_map=False, engram_table_sizes=[4])
    out = {}
    convert_layer(layer, t, cfg, {}, lambda name, w: out.__setitem__(name, w), torch.bfloat16, torch.bfloat16,
                  "bf16", table=False)
    assert t == {}  # every training tensor of the layer is consumed
    hp = "model.layers.0.mixer."
    for i in range(N - 1):
        for n in ("key_proj_extra", "key_norm_extra", "hidden_norm_extra"):
            assert hp + f"{n}.{i}.weight" in out
    assert out[hp + "gate_terms.logit_scale"].dtype == torch.float32
    assert torch.allclose(out[hp + "gate_terms.logit_scale"], torch.full((N,), 0.3), atol=2e-3)
    # the HF module's parameter names are the exported ones
    names = set(_layer(engram_branch_gates=True, engram_branch_query="learned", engram_gate_logit_scale=True)
                .mixer.state_dict())
    assert {k[len(hp):] for k in out if k.startswith(hp)} <= names

