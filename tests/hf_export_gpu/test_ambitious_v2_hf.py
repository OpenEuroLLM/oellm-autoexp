"""HF port of the ambitious v2 main run against the Megatron modules: Linear Depth Memory (stream
form), the bounded post-activation norm, quantile-balancing routing and the Engram sign hash."""

import numpy as np
import pytest
import torch
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig

# the export copies oellm_hybrid's modules next to the ambitious ones (relative imports): alias them
import sys  # noqa: E402

from oellm_autoexp.hf_export.oellm_hybrid import configuration_oellm_hybrid as _ch  # noqa: E402
from oellm_autoexp.hf_export.oellm_hybrid import modeling_oellm_hybrid as _mh  # noqa: E402

sys.modules.setdefault("oellm_autoexp.hf_export.oellm_ambitious.configuration_oellm_hybrid", _ch)
sys.modules.setdefault("oellm_autoexp.hf_export.oellm_ambitious.modeling_oellm_hybrid", _mh)

from oellm_autoexp.hf_export.oellm_ambitious.configuration_oellm_ambitious import OellmAmbitiousConfig
from oellm_autoexp.hf_export.oellm_ambitious.modeling_oellm_ambitious import (
    OellmEngramV2,
    OellmLDMBoundary,
    OellmLDMMixer,
    _swiglu,
    ldm_step,
)

from .conftest import Utils

H = 64


@pytest.fixture(autouse=True)
def mp():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(1)
    yield
    Utils.destroy_model_parallel()


def _copy(dst: torch.nn.Module, src: torch.nn.Module) -> None:
    sd = src.state_dict()
    for name, p in dst.state_dict().items():
        p.copy_(sd[name])


def _randomize(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.endswith("forget_logit"):
                p.uniform_(2.0, 5.0)  # lambda ~ 0.88-0.99, as trained
            elif name.endswith(("norm_weight", "read_gain")):
                p.uniform_(0.5, 1.5)
            else:
                p.normal_(0.0, 0.2)


def test_ldm_stream_form_matches_megatron():
    """Embedding write, three sites, the final read: HF's chain == DepthMemoryStreamBoundary +
    DepthMemoryMixer on the torch workspace (stream form, no unwind)."""
    from megatron.core.transformer.depth_memory import DepthMemoryMixer, DepthMemoryStreamBoundary
    from megatron.core.transformer.depth_memory_state import make_workspace

    g, k, hid = 4, 4, 8
    cfg = TransformerConfig(
        num_layers=3, hidden_size=H, num_attention_heads=4, layernorm_epsilon=1e-5,
        depth_memory=True, depth_memory_impl="torch", depth_memory_unwind=False,
        depth_memory_groups=g, depth_memory_key_dim=k, depth_memory_hidden_dim=hid,
        depth_memory_alpha=1.0,
    )
    torch.manual_seed(0)
    boundary = DepthMemoryStreamBoundary(cfg).cuda()
    mixers = [DepthMemoryMixer(cfg).cuda() for _ in range(3)]
    for m in [boundary, *mixers]:
        _randomize(m)
    ws = make_workspace(cfg, total_edges=4)
    boundary.wire(ws)
    for i, m in enumerate(mixers):
        m.wire(i + 1, ws)
    sub = [torch.nn.Linear(H, H, bias=False).cuda() for _ in range(3)]

    hcfg = OellmAmbitiousConfig(hidden_size=H, block_types=["moe"] * 3, residual="ldm",
                                ldm_groups=g, ldm_key_dim=k, ldm_hidden_dim=hid, ldm_alpha=1.0,
                                rms_norm_eps=1e-5)
    hb, hm = OellmLDMBoundary(hcfg).cuda(), [OellmLDMMixer(hcfg).cuda() for _ in range(3)]
    with torch.no_grad():
        _copy(hb, boundary)
        for a, b in zip(hm, mixers):
            _copy(a, b)

    emb = torch.randn(7, 2, H, device="cuda")  # [s, b, h]
    with torch.no_grad():
        streams = boundary(emb)
        ref_inputs = []
        for m, f in zip(mixers, sub):
            x_in = m.read(streams)
            ref_inputs.append(x_in)
            streams = m.write(streams, torch.tanh(f(x_in)))
        ref = boundary.collapse(streams)

        x, r, mem, cov = hb.enter(emb)
        for i, (m, f) in enumerate(zip(hm, sub)):
            x_in = x + r
            torch.testing.assert_close(x_in, ref_inputs[i], rtol=1e-4, atol=1e-5)
            out = torch.tanh(f(x_in))
            x = x + out
            r, mem, cov = ldm_step(m.write_site, m.next_query, x, out, mem, cov)
        torch.testing.assert_close(x + r, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("floor,smooth", [(1.0, False), (0.0, False), (1.0, True)])
def test_post_act_norm_matches_normed_swiglu(floor, smooth):
    from megatron.core.fusions.fused_bias_swiglu import normed_swiglu

    torch.manual_seed(1)
    gate, up = torch.randn(32, 24, device="cuda") * 3, torch.randn(32, 24, device="cuda") * 3
    ref = normed_swiglu(torch.cat([gate, up], -1), None, 10.0, 1e-5, floor, smooth)
    torch.testing.assert_close(_swiglu(gate, up, 10.0, (1e-5, floor, smooth)), ref)


def test_quantile_balancing_selection_and_weights():
    from megatron.core.transformer.moe.moe_utils import (
        qb_dual_update,
        topk_routing_with_score_function,
    )

    torch.manual_seed(2)
    logits = torch.randn(64, 12, device="cuda")
    beta = torch.randn(12, device="cuda")
    idx_ref, _ = qb_dual_update(logits, 3, beta, update_beta=False)
    probs_ref, _ = topk_routing_with_score_function(
        logits, 3, score_function="sigmoid", precomputed_indices=idx_ref
    )
    # the HF router's selection and weights (OellmMoE.forward)
    _, idx = torch.topk(logits - beta, 3, dim=-1)
    assert torch.equal(idx.sort(-1).values, idx_ref.sort(-1).values)
    w = torch.sigmoid(logits).gather(-1, idx)
    w = w / (w.sum(-1, keepdim=True) + 1e-20)
    torch.testing.assert_close(w, probs_ref.gather(-1, idx))


def test_engram_sign_hash_matches_megatron():
    from megatron.core.ssm.engram import EngramLayer

    m = dict(engram_ngram_orders=[2, 3, 4], engram_num_heads=2, engram_table_size=1000,
             engram_embed_dim=16, engram_hash_per_layer=True, engram_gate_signed_sqrt=True,
             engram_conv_prenorm=True, engram_sign_hash=True, engram_sign_table_size=64)
    cfg = TransformerConfig(num_layers=1, hidden_size=H, num_attention_heads=4,
                            layernorm_epsilon=1e-5, **m)
    pg = ProcessGroupCollection(tp=parallel_state.get_tensor_model_parallel_group(),
                                cp=parallel_state.get_context_parallel_group())
    meg = EngramLayer(cfg, layer_number=3, pg_collection=pg).cuda()
    with torch.no_grad():
        meg.table.normal_()
        meg.conv1d.weight.normal_()
        meg.hidden_norm.weight.uniform_(0.5, 1.5)
    from megatron.core.ssm.engram import engram_table_sizes

    hcfg = OellmAmbitiousConfig(
        hidden_size=H, block_types=["engram"] * 3, rms_norm_eps=1e-5,
        engram_ngram_orders=[2, 3, 4], engram_num_heads=2, engram_embed_dim=16,
        engram_table_sizes=engram_table_sizes(cfg),
        engram_hash_multipliers={"2": meg.hash_mult.tolist()},
        engram_gate_signed_sqrt=True, engram_conv_prenorm=True, engram_sign_hash=True,
        engram_sign_table_size=64, engram_sign_multipliers={"2": meg.sign_mult.tolist()},
    )
    hf = OellmEngramV2(hcfg, layer_idx=2).cuda()
    rows = sum(hcfg.engram_table_sizes)
    with torch.no_grad():
        hf.table.copy_(meg.table[:rows])
        hf.sign_table.copy_(meg.sign_table)
        for n in ("key_proj.weight", "value_proj.weight", "hidden_norm.weight", "key_norm.weight",
                  "conv1d.weight", "conv_norm.weight"):
            dict(hf.named_parameters())[n].copy_(dict(meg.named_parameters())[n])
    ids = torch.randint(0, 50000, (2, 30), device="cuda")
    h = torch.randn(30, 2, H, device="cuda")
    meg.set_input_ids(ids)
    with torch.no_grad():
        ref = meg(h)
        y, _ = hf(h.transpose(0, 1), ids)
        torch.testing.assert_close(h.transpose(0, 1) + y, ref.transpose(0, 1))
        # without the signs the outputs differ (the hash is active)
        hf.sign_hash = False
        y0, _ = hf(h.transpose(0, 1), ids)
        assert not torch.allclose(y0, y)


def test_export_sign_geometry_matches_megatron():
    """export_ambitious's sign multipliers / table == Megatron's (seeded, per layer)."""
    from megatron.core.ssm.engram import _SIGN_HASH_SEED, engram_hash_multipliers, engram_sign_table

    from oellm_autoexp.hf_export.export_ambitious import engram_sign_multipliers, engram_sign_table_np

    m = dict(engram_ngram_orders=[2, 3, 4], engram_num_heads=8, engram_hash_per_layer=True,
             engram_sign_table_size=8192, engram_embed_dim=64)
    cfg = TransformerConfig(num_layers=1, hidden_size=H, num_attention_heads=4, **m)
    for layer in (2, 51):
        assert engram_sign_multipliers(m, layer) == engram_hash_multipliers(
            cfg, layer, seed=_SIGN_HASH_SEED).tolist()
    assert np.array_equal(engram_sign_table_np(m), engram_sign_table(cfg).numpy())
