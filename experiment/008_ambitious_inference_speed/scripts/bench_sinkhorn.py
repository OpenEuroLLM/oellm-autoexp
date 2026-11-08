"""Validate and time fused mHC normalization against its eager transcription."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import torch
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
spec=importlib.util.spec_from_file_location('mhc_kernel','oellm_autoexp/hf_export/oellm_ambitious/mhc_kernels.py')
k=importlib.util.module_from_spec(spec);spec.loader.exec_module(k)

def reference(x,iters=20):
    for _ in range(iters):
        x=x-torch.logsumexp(x,-2,keepdim=True)
        x=x-torch.logsumexp(x,-1,keepdim=True)
    return x.exp()

def timed(fn,x):
    fn(x);torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(20):y=fn(x)
    torch.cuda.synchronize();return (time.perf_counter()-start)/20,y

rows=[];torch.manual_seed(42)
for n in [1,32,512,4096]:
    for scale in [0.1,1.0,10.0,100.0]:
        x=torch.randn(n,4,4,device='cuda')*scale
        old,y=timed(reference,x);new,z=timed(lambda x:k.sinkhorn4(x,20),x)
        error=float((y-z).abs().max())
        assert torch.isfinite(z).all() and error<3e-6,(n,scale,error)
        row={'tokens':n,'scale':scale,'reference_seconds':old,'fused_seconds':new,'max_abs_diff':error}
        rows.append(row);print(json.dumps(row),flush=True)
Path(a.out).write_text(json.dumps(rows,indent=2)+'\n')
