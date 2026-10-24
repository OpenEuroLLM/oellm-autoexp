"""Where KronLSTM's Megatron CP=2 output departs from CP=1 (torchrun --nproc-per-node 2):
the conv output, the op output and the layer output of each rank against the reference."""
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from fla.layers import KronLSTM
from fla.ops.kronlstm import chunk_kronlstm
from megatron.core.ssm import fla_cp

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
layer = KronLSTM(hidden_size=128, k_head_dim=32, v_head_dim=32, num_k_heads=4, num_v_heads=4).cuda()
B, S = 2, 512
x_full = torch.randn(1, B * S, 128, device="cuda")  # the flattened stream, B samples of S
group = dist.group.WORLD
ctx = fla_cp.cp_context(group, B, S, layer.conv_size, x_full.device)
n = B * S // world
sl = slice(rank * n, (rank + 1) * n)
print(f"rank {rank}: local cu {ctx.cu_seqlens.tolist()} pre {ctx.pre_num_ranks} post {ctx.post_num_ranks} "
      f"conv pre {ctx.pre_num_conv_tokens}", flush=True)
cu = torch.arange(0, (B + 1) * S, S, device="cuda", dtype=torch.long)


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


with torch.no_grad():
    qkv_full = layer.qkv_proj(x_full)
    c_ref, _ = layer.qkv_conv1d(x=qkv_full, cu_seqlens=cu)
    c_cp, _ = layer.qkv_conv1d(x=qkv_full[:, sl].contiguous(), cp_context=ctx)
    print(f"rank {rank}: conv rel err {rel(c_cp, c_ref[:, sl]):.2e}", flush=True)
    q, k, v = (F.normalize(t.view(1, -1, 4, 32), dim=-1) for t in c_ref.split((128, 128, 128), -1))
    g = -torch.rand(1, B * S, 4, 32, device="cuda") * 0.3
    beta = torch.rand(1, B * S, 4, device="cuda")
    ct = torch.rand(1, B * S, 4, device="cuda")
    o_ref, _ = chunk_kronlstm(q, k, v, g, beta, target_gate=ct, cu_seqlens=cu)
    o_cp, _ = chunk_kronlstm(*(t[:, sl].contiguous() for t in (q, k, v, g, beta)), target_gate=ct[:, sl].contiguous(),
                             cp_context=ctx)
    print(f"rank {rank}: op rel err {rel(o_cp, o_ref[:, sl]):.2e}", flush=True)
    y_ref = layer(x_full, cu_seqlens=cu)[0]
    y_dense = layer(x_full.view(B, S, 128))[0].reshape(1, B * S, 128)
    y_cp = layer(x_full[:, sl].contiguous(), cp_context=ctx)[0]
    print(f"rank {rank}: layer rel err {rel(y_cp, y_ref[:, sl]):.2e}; dense vs varlen {rel(y_dense, y_ref):.2e}",
          flush=True)
dist.destroy_process_group()
