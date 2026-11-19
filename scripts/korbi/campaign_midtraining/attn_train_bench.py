#!/usr/bin/env python3
"""Training cost of one attention layer: forward + backward of Megatron's PageSparseCoreAttention (the code the runs
train with: flex paths, block-wise selection of 64 queries) against dense flash attention, one GPU, one document of
L tokens (packed THD), v4b's shapes (hidden 2048, 16 query heads, 2 KV heads x 256), bf16. Reports ms per forward +
backward (median) and the peak memory; an out-of-memory is reported as such.

    attn_train_bench.py [--lengths 65536 131072] [--arms dense page_mean_pg page_maxpool_pg lsa_fix csa2_fix nsa]
                        [--profile N]   # also the N most expensive CUDA kernels of one timed step per arm
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F

ARMS = {  # the t16 arms' settings (page 64, top 32, window 512, 1 sink page, select_block 64, flex)
    "page_mean_pg": dict(page_sparse_selector="page_mean", page_sparse_per_group=True),
    "page_maxpool_pg": dict(page_sparse_selector="page_maxpool", page_sparse_per_group=True),
    "lsa_fix": dict(page_sparse_selector="lsa", page_sparse_lsa_pool=4, page_sparse_lsa_block=128,
                    page_sparse_lsa_buckets=8, page_sparse_lsa_visibility="block"),
    "csa2_fix": dict(page_sparse_selector="csa2", page_sparse_compress=4, page_sparse_csa2_topk_entries=512,
                     page_sparse_indexer_heads=32, page_sparse_indexer_dim=128, page_sparse_csa2_entry_bias=1.3862944,
                     page_sparse_csa2_entry_rms=True),
    "csa2_small": dict(page_sparse_selector="csa2", page_sparse_compress=4, page_sparse_csa2_topk_entries=512,
                       page_sparse_indexer_heads=4, page_sparse_indexer_dim=64, page_sparse_csa2_entry_bias=1.3862944,
                       page_sparse_csa2_entry_rms=True),
    "nsa": dict(page_sparse_selector="nsa", page_sparse_nsa_block=32, page_sparse_nsa_stride=16,
                page_sparse_nsa_select_window=512),
}
BASE = dict(page_sparse_page=64, page_sparse_topk=32, page_sparse_window=512, page_sparse_sink_pages=1,
            page_sparse_select_block=64, page_sparse_flex_q_block=64, page_sparse_backend="flex",
            page_sparse_packed_flex=True)


def timed(fn, steps=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(steps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", type=int, nargs="+", default=[65536, 131072])
    ap.add_argument("--arms", nargs="+", default=["dense", *ARMS])
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv", type=int, default=2)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--out", default=None)
    ap.add_argument("--profile", type=int, default=0, help="print the N most expensive CUDA ops of one step")
    a = ap.parse_args()

    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.experimental_attention_variant.page_sparse_attention import (
        PageSparseCoreAttention,
    )

    torch.distributed.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:29571", world_size=1, rank=0)
    parallel_state.initialize_model_parallel(1, 1)
    pg = ProcessGroupCollection(tp=parallel_state.get_tensor_model_parallel_group(),
                                cp=parallel_state.get_context_parallel_group())
    dev, dt = "cuda", torch.bfloat16
    nh, nkv, d = a.heads, a.kv, a.dim
    rows = []
    for L in a.lengths:
        for arm in a.arms:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            row = {"length": L, "arm": arm}
            try:
                q = torch.randn(L, nh, d, device=dev, dtype=dt, requires_grad=True)
                k = torch.randn(L, nkv, d, device=dev, dtype=dt, requires_grad=True)
                v = torch.randn(L, nkv, d, device=dev, dtype=dt, requires_grad=True)
                x = torch.randn(L, 1, a.hidden, device=dev, dtype=dt)
                if arm == "dense":
                    def step():
                        o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                                           v.transpose(0, 1)[None], is_causal=True,
                                                           enable_gqa=True)
                        o.sum().backward()
                else:
                    cfg = TransformerConfig(num_layers=1, hidden_size=a.hidden, num_attention_heads=nh,
                                            num_query_groups=nkv, kv_channels=d, bf16=True, **BASE, **ARMS[arm])
                    att = PageSparseCoreAttention(cfg, 1, pg_collection=pg).to(dev, dt)
                    cu = torch.tensor([0, L], dtype=torch.int32, device=dev)
                    psp = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, cu_seqlens_q_padded=cu,
                                          cu_seqlens_kv_padded=cu, max_seqlen_q=L, max_seqlen_kv=L)

                    def step():
                        att.set_hidden(x)
                        att(q, k, v, packed_seq_params=psp).sum().backward()
                row["ms"] = timed(step)
                row["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
                if a.profile:
                    from torch.profiler import ProfilerActivity, profile
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        step()
                        torch.cuda.synchronize()
                    print(f"[profile] {arm} L={L}\n" + prof.key_averages().table(
                        sort_by="cuda_time_total", row_limit=a.profile, max_name_column_width=90), flush=True)
            except torch.OutOfMemoryError:
                row["oom"] = True
            except Exception as e:  # noqa: BLE001 -- report and continue with the next arm
                row["error"] = f"{type(e).__name__}: {str(e)[:160]}"
            rows.append(row)
            print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            for name in ("q", "k", "v", "x", "att"):
                locals().pop(name, None)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"args": vars(a), "gpu": torch.cuda.get_device_name(), "rows": rows}, fh, indent=2)


if __name__ == "__main__":
    main()
