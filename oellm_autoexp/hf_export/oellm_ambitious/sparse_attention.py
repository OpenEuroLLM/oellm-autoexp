"""Sparse attention for the HF model: Megatron's ``page_sparse_selector`` (PageSparseCoreAttention) at inference.

The selection code is Megatron's own -- ``page_sparse.py`` and ``sparse_variants.py`` are copied into the export
(symlinks to submodules/Megatron-LM in this repo), so prefill and decode select exactly as training did:

* configuration: ``config.sparse_attention`` holds the run's ``page_sparse_*`` arguments under their Megatron names
  (``export_ambitious.hf_config``); unset = dense attention.
* parameters: the trained variants' (NSA gates / compressors, CSA2 compression + indexer, D-LSA latent maps, the
  learned page indexer) under ``mixer.sparse.{variant,indexer}``, exported from ``core_attention.{variant,indexer}``.
* block-wise selection (``page_sparse_select_block`` B > 1): B queries share the selection of the block's first query
  (absolute positions). A call that starts inside a block (decode, chunked prefill) gets the block's earlier queries
  prepended from the cache, runs from the block start and drops their outputs -- the same rule as Megatron's context
  parallelism (``PageSparseCoreAttention._cp_block_heads``), so any split of the sequence gives the one-shot result.
* the selector's hidden input is the attention layer's PRE-norm input (Megatron fuses the input norm into
  ``linear_qkv``; ``set_hidden`` receives the raw stream), the normalized one for a V (kvshare) layer.
* batch elements are attended one at a time (page selection is per sequence); no padding / packed documents.
"""

from __future__ import annotations

import torch
from torch import nn

# "from .<module> import ..." -- the form transformers' remote-code loader follows to copy a module next to the model
# code ("from . import page_sparse" is not followed, so the export's copy would never reach its module cache)
from .page_sparse import (PAGE_SELECTORS, PageIndexer, PageSparseConfig, dense_attention, flex_page_sparse_attention,
                          page_pool, page_sparse_attention)
from .sparse_variants import CSA2, LSA, NSA, CSA2Config, LSAConfig, NSAConfig

# a call with at least this many queries uses the flex kernels on CUDA (prefill); shorter ones (decode) the exact
# gathered reference, which needs no compilation for each new shape
FLEX_MIN_QUERIES = 1024


def _get(sa: dict, name: str, default=None):
    v = sa.get(name, default)
    return default if v is None else v


class OellmSparseAttention(nn.Module):
    """``forward(q, k, v, x, past, state) -> (out, state)``: q [b, nh, s, d] at positions past .., k / v [b, nkv, S, d]
    incl. the past (S = past + s), x [b, s, h] the selector's hidden input; out [b, nh, s, d]. ``state`` carries the
    current block's earlier queries / hidden inputs and the key side's per-token features (csa2 / indexer)."""

    def __init__(self, sa: dict, hidden: int, num_heads: int, num_kv: int, head_dim: int):
        super().__init__()
        sel = sa["page_sparse_selector"]
        self.cfg = PageSparseConfig(
            selector=sel, page=_get(sa, "page_sparse_page", 64), topk=_get(sa, "page_sparse_topk", 32),
            window=_get(sa, "page_sparse_window", 512), sink_pages=_get(sa, "page_sparse_sink_pages", 1),
            compress=_get(sa, "page_sparse_compress", 4), hier_group=_get(sa, "page_sparse_hier_group", 0),
            hier_topk=_get(sa, "page_sparse_hier_topk", 0), select_block=_get(sa, "page_sparse_select_block", 1),
            select_score=_get(sa, "page_sparse_select_score", "first"),
            per_group=bool(_get(sa, "page_sparse_per_group", False)),
        )
        self.scale = head_dim**-0.5
        self.q_block = _get(sa, "page_sparse_flex_q_block", 64)
        self.indexer = self.variant = None
        cfg = self.cfg
        if sel == "indexer":
            self.indexer = PageIndexer(hidden, _get(sa, "page_sparse_indexer_heads", 4),
                                          _get(sa, "page_sparse_indexer_dim", 64), cfg.page)
        elif sel == "nsa":
            self.variant = NSA(hidden, num_heads, head_dim, NSAConfig(
                block=_get(sa, "page_sparse_nsa_block", 32), stride=_get(sa, "page_sparse_nsa_stride", 16),
                page=cfg.page, topk=cfg.topk, sink_pages=cfg.sink_pages,
                select_window=_get(sa, "page_sparse_nsa_select_window", 512), window=cfg.window,
                select_block=cfg.select_block))
        elif sel == "csa2":
            self.variant = CSA2(hidden, num_kv, head_dim, CSA2Config(
                compress=cfg.compress, topk_entries=_get(sa, "page_sparse_csa2_topk_entries", 512),
                window=cfg.window, page=cfg.page, sink_pages=cfg.sink_pages,
                index_heads=_get(sa, "page_sparse_indexer_heads", 4),
                index_dim=_get(sa, "page_sparse_indexer_dim", 64), select_block=cfg.select_block,
                entry_bias=_get(sa, "page_sparse_csa2_entry_bias", 0.0),
                entry_rms=bool(_get(sa, "page_sparse_csa2_entry_rms", False))))
        elif sel == "lsa":
            self.variant = LSA(LSAConfig(
                pool=_get(sa, "page_sparse_lsa_pool", 4), block=_get(sa, "page_sparse_lsa_block", 512),
                buckets=_get(sa, "page_sparse_lsa_buckets", 32), topk=cfg.topk, window=cfg.window, page=cfg.page,
                sink_pages=cfg.sink_pages, select_block=cfg.select_block,
                latent=bool(_get(sa, "page_sparse_lsa_latent", False)),
                visibility=_get(sa, "page_sparse_lsa_visibility", "window")), num_kv, head_dim)
        elif sel not in PAGE_SELECTORS and sel not in ("dense", "compress_mean"):
            raise ValueError(f"sparse_attention: selector {sel!r} is not supported")
        self.uses_hidden = self.indexer is not None or sel in ("nsa", "csa2")

    def _features(self, x):
        """The key side's per-token features of the new tokens: csa2's compression weights, the indexer's keys."""
        if self.indexer is not None:
            return self.indexer.wk(x)
        if self.cfg.selector == "csa2":
            return self.variant.token_features(x)
        return None

    def _attend(self, q, k, v, q_offset, xq, feat):
        """One sequence: q [1, hq, n, d] at q_offset.., k / v [1, g, L, d], xq [1, n, h], feat [1, L, F]."""
        sel, cfg = self.cfg.selector, self.cfg
        flex = q.is_cuda and q.shape[2] >= FLEX_MIN_QUERIES
        if sel == "dense":
            return dense_attention(q, k, v, q_offset, self.scale)
        if self.variant is not None:
            self.variant.flex = flex
            if sel == "nsa":
                return self.variant(q, k, v, xq, q_offset, self.scale)
            if sel == "csa2":
                return self.variant(q, k, v, xq, feat, q_offset, self.scale)
            return self.variant(q, k, v, q_offset, self.scale)
        scores = None
        if self.indexer is not None:
            kp = page_pool(feat.unsqueeze(1), cfg.page, "mean")[:, 0]
            scores = self.indexer(xq, kp).detach()
        if flex:
            return flex_page_sparse_attention(q, k, v, cfg, q_offset, self.scale, scores, self.q_block)
        return page_sparse_attention(q, k, v, cfg, q_offset, self.scale, scores)

    def forward(self, q, k, v, x, past: int, state=None):
        b, s = q.shape[0], q.shape[2]
        B = max(1, int(self.cfg.select_block))
        st = state or {}
        h = past % B if past > 0 else 0  # the block's earlier queries, prepended
        if h:
            q = torch.cat([st["qh"].to(q.dtype), q], 2)
            if self.uses_hidden:
                x = torch.cat([st["xh"].to(x.dtype), x], 1)
        new = self._features(x[:, h:]) if (self.uses_hidden and x is not None) else None
        feat = None
        if new is not None:
            feat = new if st.get("feat") is None or past == 0 else torch.cat([st["feat"], new], 1)
        outs = []
        for i in range(b):
            xi = None if x is None else x[i:i + 1]
            fi = None if feat is None else feat[i:i + 1]
            outs.append(self._attend(q[i:i + 1], k[i:i + 1], v[i:i + 1], past - h, xi, fi))
        out = torch.cat(outs, 0)[:, :, h:]
        r = (past + s) % B  # the queries of the block the next call continues
        new_state = {"qh": q[:, :, q.shape[2] - r:].detach() if r else q[:, :, :0].detach(),
                     "xh": (x[:, x.shape[1] - r:].detach() if r else x[:, :0].detach()) if self.uses_hidden else None,
                     "feat": feat}
        return out, new_state
