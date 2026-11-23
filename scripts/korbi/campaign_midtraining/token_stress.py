#!/usr/bin/env python3
"""Stress test of the per-token sparse attention path as the v4b training runs it: several attention layers
(PageSparseCoreAttention, THD, page_maxpool per KV group, select_block 1, rowmask) per step, each under activation
checkpointing (forward without grad, recompute in the backward, layers in reverse), over many real cross-document
layouts (cds_spans json), with a side stream allocating / freeing memory concurrently (FSDP-like allocator traffic).
Checks every step that outputs and gradients are finite, and that a repeated step with the same inputs gives
bit-identical results (a race or an uninitialized read shows up as a mismatch or an illegal address). Run under
CUDA_LAUNCH_BLOCKING=1 and/or compute-sanitizer (--steps small) to localize faults.

    token_stress.py --spans cds_spans.json [--steps 200 --layers 4 --heads 16 --kv 2 --dim 128 --repeat 2]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spans", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv", type=int, default=2)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=2, help="run each step this often with the same inputs; compare")
    ap.add_argument("--seq", type=int, default=0, help="cut layouts to this many tokens (0: as given)")
    ap.add_argument("--no-side", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    import torch.distributed as dist
    from torch.utils.checkpoint import checkpoint

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29600 + a.seed))
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)
    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.experimental_attention_variant.page_sparse_attention import (
        PageSparseCoreAttention,
    )

    parallel_state.initialize_model_parallel(1, 1)
    pg = ProcessGroupCollection(tp=parallel_state.get_tensor_model_parallel_group(),
                                cp=parallel_state.get_context_parallel_group())
    HQ, G, D = a.heads, a.kv, a.dim
    cfg = TransformerConfig(num_layers=a.layers, hidden_size=HQ * D, num_attention_heads=HQ, num_query_groups=G,
                            kv_channels=D, page_sparse_selector="page_maxpool", page_sparse_per_group=True,
                            page_sparse_select_block=1, page_sparse_backend="rowmask", page_sparse_page=64,
                            page_sparse_topk=32, page_sparse_window=512, page_sparse_sink_pages=1,
                            page_sparse_flex_q_block=64, page_sparse_packed_flex=True, bf16=True)
    layers = [PageSparseCoreAttention(cfg, i + 1, pg_collection=pg).cuda() for i in range(a.layers)]
    seqs = json.load(open(a.spans))["seqs"]
    rng = random.Random(a.seed)

    stop = threading.Event()

    def side():  # concurrent allocator traffic on another stream (FSDP all-gathers / reduce-scatters)
        s = torch.cuda.Stream()
        r = random.Random(1)
        with torch.cuda.stream(s):
            while not stop.is_set():
                n = r.choice([1, 4, 16, 64, 256]) << 18
                x = torch.empty(n, device="cuda", dtype=torch.bfloat16).normal_()
                y = x * 2
                del x, y
                time.sleep(0.0005)

    th = None
    if not a.no_side:
        th = threading.Thread(target=side, daemon=True)
        th.start()

    def step(segs, gen_seed, ckpt=True):
        L = sum(segs)
        cu = torch.tensor([0] + list(torch.tensor(segs).cumsum(0).tolist()), dtype=torch.int32, device="cuda")
        psp = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, cu_seqlens_q_padded=cu,
                              cu_seqlens_kv_padded=cu, max_seqlen_q=max(segs), max_seqlen_kv=max(segs))
        g = torch.Generator(device="cuda").manual_seed(gen_seed)

        def mk(n):
            x = 0.3 * torch.randn(L, n, D, generator=g, device="cuda")
            x = x + torch.randn(L // 256 + 1, n, D, generator=g, device="cuda").repeat_interleave(256, 0)[:L]
            return x.to(torch.bfloat16).requires_grad_()

        ins = [(mk(HQ), mk(G), mk(G)) for _ in layers]
        outs = []
        for att, (q, k, v) in zip(layers, ins):
            fn = lambda q_, k_, v_, att=att: att(q_, k_, v_, packed_seq_params=psp)
            outs.append(checkpoint(fn, q, k, v, use_reentrant=False) if ckpt else fn(q, k, v))
        loss = sum(o.float().square().mean() for o in outs)
        loss.backward()
        grads = [t.grad for qkv in ins for t in qkv]
        return [o.detach() for o in outs], grads

    step(seqs[0], 0, ckpt=False)  # the first call warms the kernels up (its own fwd + bwd) -- outside a checkpoint
    torch.cuda.synchronize()
    t0 = time.time()
    bad = 0
    for it in range(a.steps):
        segs = seqs[rng.randrange(len(seqs))]
        if a.seq:
            cut, acc = [], 0
            for s_ in segs:
                if acc + s_ >= a.seq:
                    cut.append(a.seq - acc)
                    break
                cut.append(s_)
                acc += s_
            segs = cut
        res = []
        for r in range(a.repeat):
            o, gr = step(segs, 1000 + it)
            torch.cuda.synchronize()
            fin = all(bool(torch.isfinite(x).all()) for x in o + gr)
            res.append((o, gr, fin))
        same = all(all(torch.equal(x, y) for x, y in zip(res[0][0] + res[0][1], r_[0] + r_[1])) for r_ in res[1:])
        ok = all(r_[2] for r_ in res) and same
        bad += not ok
        if not ok or it % 20 == 0:
            print(json.dumps(dict(it=it, spans=len(segs), longest=max(segs), finite=[r_[2] for r_ in res],
                                  deterministic=same, t=round(time.time() - t0, 1))), flush=True)
    stop.set()
    print("STRESS DONE", json.dumps(dict(steps=a.steps, bad=bad, seconds=round(time.time() - t0, 1))), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
