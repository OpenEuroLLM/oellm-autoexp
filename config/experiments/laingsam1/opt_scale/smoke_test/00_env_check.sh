#!/bin/bash
# Environment check, run on a JUPITER LOGIN node (they have a GH200). Uses no GPU hours.
#   bash config/experiments/laingsam1/opt_scale/smoke_test/00_env_check.sh
# Checks, inside the same container + PYTHONPATH the jobs use:
#   - megatron.core comes from the fork (not the container's bundled 0.17)
#   - emerging_optimizers comes from pyext (0.2.0) and the fork sees it
#     (HAVE_EMERGING_OPTIMIZERS, AngularMuown class present)
#   - fla is usable (HAVE_FLA) and one tiny Gated DeltaNet kernel runs on the GPU
# Triton compiles kernels into /tmp/$USER/triton_cache (nothing written into the repo).
set -euo pipefail

BASE=/e/scratch/e-aif-2026sc04-211/users/slaing
SIF=/e/project1/e-ext-2025e02-108/container/nemo_26.04.sif
PYTHONPATH_JOBS="$BASE/pyext:$BASE/NVIDIA-Megatron-LM-angular"

apptainer exec --nv \
  --env PYTHONPATH="$PYTHONPATH_JOBS" \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env TRITON_CACHE_DIR="/tmp/$USER/triton_cache" \
  "$SIF" /opt/venv/bin/python - <<'EOF'
import torch, torch.nn.functional as F
import megatron.core, emerging_optimizers, fla, transformer_engine

print("torch               ", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
print("transformer_engine  ", transformer_engine.__version__)
print("fla                 ", fla.__version__)
print("megatron.core from  ", megatron.core.__file__)
print("emerging_optimizers ", emerging_optimizers.__file__)

from megatron.core.optimizer import emerging_optimizers as eo
from megatron.core.ssm.gated_delta_net import HAVE_FLA
checks = {
    "megatron.core is the fork": "NVIDIA-Megatron-LM-angular" in megatron.core.__file__,
    "emerging_optimizers from pyext": "/pyext/" in emerging_optimizers.__file__,
    "HAVE_EMERGING_OPTIMIZERS": eo.HAVE_EMERGING_OPTIMIZERS,
    "AngularMuown available": hasattr(eo, "TensorParallelAngularMuown"),
    "HAVE_FLA": HAVE_FLA,
}

# One tiny Gated DeltaNet forward on the GPU (layout [B, T, H, D]).
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
torch.manual_seed(0)
B, T, H, D = 1, 256, 4, 128
dev, dt = "cuda", torch.bfloat16
q = F.normalize(torch.randn(B, T, H, D, device=dev, dtype=torch.float32), dim=-1).to(dt)
k = F.normalize(torch.randn(B, T, H, D, device=dev, dtype=torch.float32), dim=-1).to(dt)
v = torch.randn(B, T, H, D, device=dev, dtype=dt)
g = F.logsigmoid(torch.randn(B, T, H, device=dev, dtype=torch.float32))   # log decay < 0
beta = torch.rand(B, T, H, device=dev, dtype=dt).sigmoid()
out = chunk_gated_delta_rule(q, k, v, g, beta)
o = out[0] if isinstance(out, tuple) else out
checks["GDN kernel output finite"] = bool(torch.isfinite(o.float()).all())
print("GDN kernel output   ", tuple(o.shape), o.dtype)

print()
for name, ok in checks.items():
    print(f"  [{'OK' if ok else 'FAIL'}] {name}")
if not all(checks.values()):
    raise SystemExit("ENV CHECK FAILED")
print("\nENV CHECK PASSED")
EOF
