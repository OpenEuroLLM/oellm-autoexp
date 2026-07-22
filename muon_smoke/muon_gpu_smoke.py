"""Self-contained Muon smoke test on ROCm GPU.

Exercises the core Muon operation (Newton-Schulz orthogonalization) on a GPU
tensor, then confirms Megatron v0.17 detects the package. No data / no real
distributed comm (uses a trivial 1-rank group; tp_mode='duplicated').
Run inside the container+overlay on a GPU allocation. See INSTRUCTIONS.md.
"""
import os
import inspect
import torch
import torch.distributed as dist

print("=" * 60)
print("torch:", torch.__version__, "| cuda(rocm) available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "No GPU visible — did you run under srun on the allocation?"
torch.cuda.set_device(0)
print("device:", torch.cuda.get_device_name(0))

# newton_schulz_tp is tensor-parallel-aware and requires a process group.
# For a 1-GPU smoke test we make a trivial single-rank group; with
# tp_mode='duplicated' no actual cross-rank communication happens.
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29591")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
dist.init_process_group(backend="nccl", rank=0, world_size=1)  # nccl -> rccl on ROCm
group = dist.group.WORLD
print("single-rank process group: OK")

# --- 1) core Muon op: Newton-Schulz orthogonalization on GPU -------------
from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz_tp
print("newton_schulz_tp signature:", inspect.signature(newton_schulz_tp))

G = torch.randn(512, 256, device="cuda", dtype=torch.float32)  # NS runs in fp32
O = newton_schulz_tp(G, 5, "quintic", group, partition_dim=None, tp_mode="duplicated")
assert torch.isfinite(O).all(), "Newton-Schulz produced non-finite values on ROCm!"

# orthogonalized matrix -> singular values should cluster near 1
s = torch.linalg.svdvals(O.float())
print(f"  output singular values: min={s.min():.3f} max={s.max():.3f} (near 1.0 = orthogonalized)")
print("  >>> Newton-Schulz on GPU: OK")

# --- 2) Megatron v0.17 integration: does it see the package? -------------
from megatron.core.optimizer.emerging_optimizers import (
    HAVE_EMERGING_OPTIMIZERS, get_supported_coefficient_types,
)
print("HAVE_EMERGING_OPTIMIZERS:", HAVE_EMERGING_OPTIMIZERS)
print("supported coefficient types:", get_supported_coefficient_types())
assert HAVE_EMERGING_OPTIMIZERS, "Megatron cannot see emerging_optimizers"

dist.destroy_process_group()
print("=" * 60)
print("MUON GPU SMOKE: PASS")
