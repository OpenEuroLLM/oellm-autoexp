#!/bin/bash
# Environment check for smoke test 05 (team Megatron oellm/v0.19-hybrid + te218 container).
# Run on a JUPITER LOGIN node (they have a GH200). Uses no GPU hours.
#   bash config/experiments/laingsam1/opt_scale/smoke_test/05_hybrid_env_check.sh
# Checks, inside the same container + PYTHONPATH the 05 jobs use:
#   - the Megatron checkout is at the pinned commit and clean
#   - megatron.core comes from that checkout (not the container's bundled 0.17)
#   - Transformer Engine is 2.18 and its GroupedLinear accepts single_grouped_weight
#     (the reason the old fork needed a patch on nemo_26.04)
#   - emerging_optimizers comes from pyext (0.2.0); Muon + AngularMuown available
#   - fla is usable (HAVE_FLA) and one tiny Gated DeltaNet kernel runs on the GPU
# Read-only apart from Triton's kernel cache in /tmp/$USER/triton_cache.
set -euo pipefail

BASE=${OPT_SCALE_BASE:-/e/scratch/e-aif-2026sc04-211/users/slaing}   # see opt_scale/README.md
SIF=/e/project1/e-sta-openeurollm/container/MegatronTraining-JUPITER-te218-fa3_aarch64_202608280932.sif
MEG=$BASE/NVIDIA-Megatron-LM-hybrid
PIN=013763961102c13bc8d34ee501e616429aa3e5c1

head=$(git -C "$MEG" rev-parse HEAD)
dirty=$(git -C "$MEG" status --porcelain --untracked-files=no | wc -l)
echo "Megatron checkout   $MEG @ ${head:0:9} (modified tracked files: $dirty)"
[ "$head" = "$PIN" ] || { echo "  [FAIL] expected pinned commit ${PIN:0:9}"; exit 1; }
[ "$dirty" -eq 0 ] || { echo "  [FAIL] checkout has local modifications"; exit 1; }

apptainer exec --nv \
  --env PYTHONPATH="$BASE/pyext:$MEG" \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env TRITON_CACHE_DIR="/tmp/$USER/triton_cache" \
  "$SIF" /opt/venv/bin/python - <<'EOF'
import inspect
import torch, torch.nn.functional as F
import megatron.core, emerging_optimizers, fla, transformer_engine
import transformer_engine.pytorch as te

print("torch               ", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
print("transformer_engine  ", transformer_engine.__version__)
print("fla                 ", fla.__version__)
print("megatron.core from  ", megatron.core.__file__)
print("emerging_optimizers ", emerging_optimizers.__file__)

from megatron.core.optimizer import emerging_optimizers as eo
from megatron.core.ssm.gated_delta_net import HAVE_FLA
checks = {
    "megatron.core is the hybrid checkout": "NVIDIA-Megatron-LM-hybrid" in megatron.core.__file__,
    "TE >= 2.18": tuple(int(x) for x in transformer_engine.__version__.split("+")[0].split(".")[:2]) >= (2, 18),
    "TE GroupedLinear takes single_grouped_weight":
        "single_grouped_weight" in inspect.signature(te.GroupedLinear.__init__).parameters,
    "emerging_optimizers from pyext": "/pyext/" in emerging_optimizers.__file__,
    "HAVE_EMERGING_OPTIMIZERS": eo.HAVE_EMERGING_OPTIMIZERS,
    "Muon available": hasattr(eo, "TensorParallelMuon"),
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
