"""Sparse attention in the HF model (oellm_ambitious/sparse_attention.py): Megatron's page_sparse_selector at inference.

float64 on CPU, so every selector runs Megatron's exact gathered reference path:
* a full budget is dense attention;
* the module equals Megatron's page_sparse_attention on its own q / k / v;
* block-wise selection survives any split of the sequence: chunked prefill at unaligned offsets and token-by-token
  decode give the one-shot outputs (the block's earlier queries come from the cache), for the page selector and the
  trained variants (NSA / CSA2 / LSA);
* the selector reads the pre-norm hidden input (``raw``), not the normalized one;
* a V (kvshare) layer selects with its own queries over its source's keys, also across decode steps.
"""
import sys

import pytest
import torch

from oellm_autoexp.hf_export.oellm_hybrid import configuration_oellm_hybrid as _ch
from oellm_autoexp.hf_export.oellm_hybrid import modeling_oellm_hybrid as _mh

sys.modules.setdefault("oellm_autoexp.hf_export.oellm_ambitious.configuration_oellm_hybrid", _ch)
sys.modules.setdefault("oellm_autoexp.hf_export.oellm_ambitious.modeling_oellm_hybrid", _mh)

from oellm_autoexp.hf_export.oellm_ambitious import page_sparse as ps  # noqa: E402
from oellm_autoexp.hf_export.oellm_ambitious.configuration_oellm_ambitious import (  # noqa: E402
    OellmAmbitiousConfig,
)
from oellm_autoexp.hf_export.oellm_ambitious.modeling_oellm_ambitious import (  # noqa: E402
    OellmKVShareAttention,
    OellmXSAAttention,
)

HID, NH, NKV, D, L = 32, 4, 2, 8, 70  # 70 tokens: 8 full select blocks of 8 and a partial one

BASE = dict(page_sparse_page=8, page_sparse_topk=2, page_sparse_window=8, page_sparse_sink_pages=1,
            page_sparse_select_block=8)
VARIANTS = {
    "page_mean_pg": dict(page_sparse_selector="page_mean", page_sparse_per_group=True),
    "page_maxpool_pg": dict(page_sparse_selector="page_maxpool", page_sparse_per_group=True),
    "nsa": dict(page_sparse_selector="nsa", page_sparse_nsa_block=8, page_sparse_nsa_stride=4,
                page_sparse_nsa_select_window=8),
    "csa2": dict(page_sparse_selector="csa2", page_sparse_compress=2, page_sparse_csa2_topk_entries=8,
                 page_sparse_indexer_heads=2, page_sparse_indexer_dim=8, page_sparse_csa2_entry_bias=1.386,
                 page_sparse_csa2_entry_rms=True),
    "lsa": dict(page_sparse_selector="lsa", page_sparse_lsa_pool=2, page_sparse_lsa_block=8,
                page_sparse_lsa_buckets=4, page_sparse_lsa_visibility="block"),
}


def _cfg(sparse=None, **kw):
    return OellmAmbitiousConfig(
        vocab_size=64, hidden_size=HID, num_hidden_layers=2, num_attention_heads=NH, head_dim=D,
        num_key_value_heads=NKV, block_types=["attention", "kvshare"], kv_share_sources={"1": 0},
        residual="mhc", mhc_expansion_rate=4, rms_norm_eps=1e-5, qk_layernorm=True, attention_output_gate=True,
        exclusive_self_attention=True, sparse_attention=sparse, **kw)


def _attn(sparse):
    torch.manual_seed(0)
    att = OellmXSAAttention(_cfg(sparse)).double()
    for p in att.parameters():  # random, not the identity / zero init: every parameter matters
        torch.nn.init.normal_(p, std=0.3)
    return att


def _inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, L, HID, generator=g, dtype=torch.float64)
    raw = torch.randn(1, L, HID, generator=g, dtype=torch.float64)
    return x, raw


def _run_split(att, x, raw, cuts):
    """The module over x in pieces ending at ``cuts`` (exclusive), carrying the cache state."""
    outs, state, a = [], None, 0
    for c in list(cuts) + [L]:
        o, state = att(x[:, a:c], state, True, raw=raw[:, a:c])
        outs.append(o)
        a = c
    return torch.cat(outs, 1)


def test_full_budget_is_dense():
    x, raw = _inputs()
    dense = _attn(None)
    full = _attn(dict(page_sparse_selector="page_mean", page_sparse_page=8, page_sparse_topk=64,
                      page_sparse_window=64, page_sparse_sink_pages=1, page_sparse_select_block=8))
    full.load_state_dict(dense.state_dict())
    torch.testing.assert_close(full(x, None, True, raw=raw)[0], dense(x, None, True)[0], atol=1e-10, rtol=1e-10)


def test_module_is_megatrons_selection():
    """The sparse output before XSA / gate / o_proj equals page_sparse_attention on the same q / k / v."""
    att = _attn(dict(BASE, **VARIANTS["page_mean_pg"]))
    x, raw = _inputs()
    seen = {}
    inner = att.sparse.forward

    def capture(q, k, v, xx, past, state=None):
        out, st = inner(q, k, v, xx, past, state)
        seen.update(q=q, k=k, v=v, out=out)
        return out, st

    att.sparse.forward = capture
    att(x, None, True, raw=raw)
    ref = ps.page_sparse_attention(seen["q"], seen["k"], seen["v"], att.sparse.cfg, 0, D**-0.5)
    torch.testing.assert_close(seen["out"], ref, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("name", list(VARIANTS))
@pytest.mark.parametrize("cuts", [(37,), (5, 13, 29, 30, 51), tuple(range(30, L))],
                         ids=["chunked", "ragged", "decode"])
def test_any_split_gives_the_one_shot_output(name, cuts):
    att = _attn(dict(BASE, **VARIANTS[name]))
    x, raw = _inputs()
    one = att(x, None, True, raw=raw)[0]
    torch.testing.assert_close(_run_split(att, x, raw, cuts), one, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("name", ["nsa", "csa2"])
def test_selector_reads_the_pre_norm_input(name):
    """NSA / CSA2 select from the raw stream (Megatron's set_hidden), so a different raw input changes the output
    while q / k / v (from the normalized input) stay the same; without ``raw`` the normalized input is used."""
    att = _attn(dict(BASE, **VARIANTS[name]))
    x, raw = _inputs()
    a = att(x, None, True, raw=raw)[0]
    b = att(x, None, True, raw=raw * 3.0 + 1.0)[0]
    c = att(x, None, True, raw=x)[0]
    torch.testing.assert_close(att(x, None, True)[0], c, atol=0, rtol=0)
    assert (a - b).abs().max() > 1e-6


def test_kvshare_layer_selects_over_its_sources_keys():
    cfg = _cfg(dict(BASE, **VARIANTS["page_mean_pg"]))
    torch.manual_seed(0)
    src, vl = OellmXSAAttention(cfg).double(), OellmKVShareAttention(cfg).double()
    for p in list(src.parameters()) + list(vl.parameters()):
        torch.nn.init.normal_(p, std=0.3)
    src.keep_kv = True
    x, _ = _inputs()
    src(x, None, True)
    vl.kv = src.last_kv
    one = vl(x, None, True)[0]
    outs, s_state, v_state, a = [], None, None, 0
    for c in (33, 34, 47, L):
        _, s_state = src(x[:, a:c], s_state, True)
        vl.kv = src.last_kv
        o, v_state = vl(x[:, a:c], v_state, True)
        outs.append(o)
        a = c
    torch.testing.assert_close(torch.cat(outs, 1), one, atol=1e-10, rtol=1e-10)
