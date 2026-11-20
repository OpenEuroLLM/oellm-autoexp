"""Peak memory and time of one Engram layer's forward + backward under mHC at the Nucleus width (hidden 2048,
4 streams, orders 2-4 x 8 heads x 64, sign hash, signed-sqrt gate, conv prenorm; small table), bf16: the single
gate against per-stream gates (engram_branch_gates, query read / stream). Run under torchrun, 1 GPU."""

import os
import sys
import time

import torch

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.engram import EngramLayer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig

SEQ = int(os.environ.get("SEQ", "65536"))
HIDDEN, N = 2048, 4


def layer(pg, **kw):
    cfg = TransformerConfig(
        num_layers=1, hidden_size=HIDDEN, num_attention_heads=16, params_dtype=torch.bfloat16, bf16=True,
        engram_ngram_orders=[2, 3, 4], engram_num_heads=8, engram_table_size=20011, engram_embed_dim=64,
        engram_gate_signed_sqrt=True, engram_conv_prenorm=True, engram_sign_hash=True,
        mhc=True, mhc_expansion_rate=N, mhc_sinkhorn_iters=20, **kw,
    )
    return EngramLayer(cfg, layer_number=1, pg_collection=pg).cuda().bfloat16()


def measure(m, ids, x, reps=3):
    m.set_input_ids(ids)
    for p in m.parameters():
        p.grad = None
    m(x).float().square().mean().backward()  # warm-up (compile)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.time()
    for _ in range(reps):
        m(x).float().square().mean().backward()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**30, (time.time() - t0) / reps * 1e3


def main():
    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(0)
    parallel_state.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(0)
    pg = ProcessGroupCollection(tp=parallel_state.get_tensor_model_parallel_group(),
                                cp=parallel_state.get_context_parallel_group())
    ids = torch.randint(0, 262144, (1, SEQ), device="cuda")
    x = torch.randn(SEQ, 1, N, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    print(f"seq {SEQ}, hidden {HIDDEN}, {N} streams; stream state {x.numel() * 2 / 2**30:.2f} GiB")
    for name, kw in (("single", {}), ("read", dict(engram_branch_gates=True)),
                     ("stream", dict(engram_branch_gates=True, engram_branch_query="stream"))):
        gib, ms = measure(layer(pg, **kw), ids, x)
        print(f"{name:7s} peak {gib:6.2f} GiB over the inputs, {ms:7.1f} ms fwd+bwd", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
