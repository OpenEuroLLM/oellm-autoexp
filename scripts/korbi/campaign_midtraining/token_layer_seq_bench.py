#!/usr/bin/env python3
"""One v4b attention layer (16 / 2 heads, head dim 256) through the full PageSparseCoreAttention THD path (packing,
selection, kernels, output glue) or dense flash-attention varlen, forward + backward over a SEQUENCE of real packed
layouts in ONE process -- to catch what a fresh-process benchmark misses: growth of the allocator pool or of non-PyTorch
device memory (local memory of spilled kernels) across varying layouts, and the per-layout time spread. Layouts: the
cross-document plan's group spans (cds_spans json: {"seq": 65536, "seqs": [[span, ...], ...]}), the first --n plus the
--long longest-span ones. q / k get a smooth component (neighbouring tokens alike) so selections overlap roughly as
trained ones do (--rough: plain random, the scattered worst case). Per iteration: ms fwd / bwd, PyTorch peak above the
inputs, reserved pool, non-PyTorch device memory (device used - reserved), all in GB.

    token_layer_seq_bench.py --spans cds_spans.json --variant tok|sb16|dense [--n 30 --long 10 --passes 2] [--out json]
"""

from __future__ import annotations

import argparse
import json
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spans", required=True)
    ap.add_argument("--variant", default="tok")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--long", type=int, default=10)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--rough", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)
    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer import TransformerConfig

    parallel_state.initialize_model_parallel(1, 1)
    pg = ProcessGroupCollection(tp=parallel_state.get_tensor_model_parallel_group(),
                                cp=parallel_state.get_context_parallel_group())
    HQ, G, D, H = 16, 2, 256, 2048
    d = json.load(open(a.spans))
    seqs = d["seqs"]
    pick = list(range(min(a.n, len(seqs))))
    longest = sorted(range(len(seqs)), key=lambda i: -max(seqs[i]))[: a.long]
    pick += [i for i in longest if i not in pick]

    att = None
    if a.variant != "dense":
        from megatron.core.transformer.experimental_attention_variant.page_sparse_attention import (
            PageSparseCoreAttention,
        )

        sb = {"tok": 1, "sb16": 16, "sb8": 8}[a.variant]
        cfg = TransformerConfig(num_layers=1, hidden_size=H, num_attention_heads=HQ, num_query_groups=G, kv_channels=D,
                                page_sparse_selector="page_maxpool", page_sparse_per_group=True,
                                page_sparse_select_block=sb, page_sparse_backend="rowmask", page_sparse_page=64,
                                page_sparse_topk=32, page_sparse_window=512, page_sparse_sink_pages=1,
                                page_sparse_flex_q_block=64, page_sparse_packed_flex=True, bf16=True)
        att = PageSparseCoreAttention(cfg, 1, pg_collection=pg).cuda()

    def make(L, gen):
        def one(n):
            x = torch.randn(L, n, D, generator=gen, device="cuda", dtype=torch.float32)
            if not a.rough:  # a smooth component: neighbouring tokens alike (overlapping selections)
                s = torch.randn(L // 256 + 1, n, D, generator=gen, device="cuda").repeat_interleave(256, 0)[:L]
                x = 0.3 * x + s
            return x.to(torch.bfloat16).requires_grad_()
        return one(HQ), one(G), one(G)

    rows = []
    gen = torch.Generator(device="cuda").manual_seed(0)
    torch.cuda.synchronize()
    free_base = torch.cuda.mem_get_info()[0]
    for p in range(a.passes):
        for i in pick:
            segs = seqs[i]
            L = sum(segs)
            cu = torch.tensor([0] + list(torch.tensor(segs).cumsum(0).tolist()), dtype=torch.int32, device="cuda")
            q, k, v = make(L, gen)
            torch.cuda.synchronize()
            base_alloc = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            if att is None:
                from flash_attn import flash_attn_varlen_func

                mx = max(segs)
                o = flash_attn_varlen_func(q, k, v, cu, cu, mx, mx, causal=True, softmax_scale=D**-0.5)
                o = o.reshape(L, HQ * D)
            else:
                psp = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, cu_seqlens_q_padded=cu,
                                      cu_seqlens_kv_padded=cu, max_seqlen_q=max(segs), max_seqlen_kv=max(segs))
                o = att(q, k, v, packed_seq_params=psp)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            o.float().sum().backward()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            peak = torch.cuda.max_memory_allocated() - base_alloc
            res = torch.cuda.memory_reserved()
            free = torch.cuda.mem_get_info()[0]
            nontorch = (free_base - free) - res
            row = dict(p=p, seq=i, spans=len(segs), longest=max(segs), fwd_ms=1e3 * (t1 - t0), bwd_ms=1e3 * (t2 - t1),
                       peak_gb=peak / 2**30, reserved_gb=res / 2**30, nontorch_gb=nontorch / 2**30)
            rows.append(row)
            print(json.dumps({k_: (round(v_, 3) if isinstance(v_, float) else v_) for k_, v_ in row.items()}), flush=True)
            del q, k, v, o
    last = [r for r in rows if r["p"] == a.passes - 1]
    summ = dict(variant=a.variant, rough=a.rough,
                ms_mean=sum(r["fwd_ms"] + r["bwd_ms"] for r in last) / len(last),
                ms_max=max(r["fwd_ms"] + r["bwd_ms"] for r in last),
                peak_gb_max=max(r["peak_gb"] for r in rows), reserved_gb_end=rows[-1]["reserved_gb"],
                reserved_gb_max=max(r["reserved_gb"] for r in rows),
                nontorch_gb_first=rows[0]["nontorch_gb"], nontorch_gb_end=rows[-1]["nontorch_gb"])
    print("SUMMARY", json.dumps({k_: (round(v_, 3) if isinstance(v_, float) else v_) for k_, v_ in summ.items()}),
          flush=True)
    if a.out:
        json.dump({"rows": rows, "summary": summ}, open(a.out, "w"), indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
