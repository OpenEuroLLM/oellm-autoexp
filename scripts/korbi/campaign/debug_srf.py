"""Debug the stable-rank floor inside TensorParallelMuownClassical (cluster: needs emerging_optimizers)."""
import torch
import megatron.core.optimizer.emerging_optimizers as eo

calls = {"n": 0, "capped": 0}
orig = eo._stable_rank_floor_
def wrapped(w, r_min, *a, **k):
    calls["n"] += 1
    c = orig(w, r_min, *a, **k)
    calls["capped"] += c
    return c
eo._stable_rank_floor_ = wrapped

def srank(w):
    s = torch.linalg.svdvals(w.detach().float()); return float(s.square().sum() / s[0].square())

for batched in (False, True):
    torch.manual_seed(0)
    expert = torch.nn.Parameter(torch.randn(48, 96, device='cuda') * 0.02); expert.allreduce = False
    dense = torch.nn.Parameter(expert.detach().clone())
    opt = eo.TensorParallelMuownClassical(params=[expert, dense], lr=0.05, momentum=0.95, nesterov=True,
        betas=(0.9, 0.95), num_ns_steps=5, scale_mode="shape_scaling", pg_collection=None,
        tp_mode="duplicated", batched_step=batched, stable_rank_floor=10.0)
    print("batched", batched, "floor attr", opt._stable_rank_floor, "row reduce group", opt._tp_row_reduce_group(expert))
    d = torch.randn(48, 1, device='cuda') @ torch.randn(1, 96, device='cuda')
    calls.update(n=0, capped=0)
    for i in range(300):
        for p in (expert, dense):
            p.grad = -d.clone()
        opt.step()
        if i in (0, 10, 50, 299):
            print(f"  step {i}: srank expert {srank(expert):.2f} dense {srank(dense):.2f} calls {calls}")
