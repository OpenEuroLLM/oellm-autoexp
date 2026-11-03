"""Real checkpoint MoE: reference, cached dequantization and grouped-mm probe."""
import argparse
import importlib
import json
from pathlib import Path
import time

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--out',required=True)
a=p.parse_args();root=Path(a.model)
cfg=AutoConfig.from_pretrained(root,trust_remote_code=True)
layer=cfg.block_types.index('moe');prefix=f'model.layers.{layer}.mixer.'
with torch.device('meta'):
    full=AutoModelForCausalLM.from_config(cfg,trust_remote_code=True,dtype=torch.bfloat16)
mod=full.model.layers[layer].mixer
del full
index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
groups={}
for name,file in index.items():
    if name.startswith(prefix):groups.setdefault(file,[]).append(name)
sd={}
for file,names in groups.items():
    with safe_open(root/file,framework='pt',device='cpu') as f:
        for name in names:sd[name[len(prefix):]]=f.get_tensor(name)
mod.load_state_dict(sd,assign=True);mod=mod.to('cuda').eval();del sd
act=importlib.import_module(type(mod).__module__)._swiglu

def timed(fn,x):
    with torch.no_grad():
        fn(x);torch.cuda.synchronize();started=time.perf_counter()
        for _ in range(2):out=fn(x)
        torch.cuda.synchronize()
    return (time.perf_counter()-started)/2,out

torch.manual_seed(42)
xs={n:torch.randn(n,cfg.hidden_size,device='cuda',dtype=torch.bfloat16) for n in [1,128,512]}
results={'layer':layer,'torch':torch.__version__,'trials':[]}
refs={};times={}
for n,x in xs.items():
    times[n],refs[n]=timed(mod,x)
    print(json.dumps({'phase':'reference','tokens':n,'seconds':times[n]}),flush=True)
started=time.perf_counter()
if mod.fp8:
    for name in ['gate_up_proj','down_proj']:
        q=getattr(mod,name).view(torch.float8_e4m3fn).float()
        sc=getattr(mod,name+'_scale').float()
        value=(q.view(*q.shape[:-1],-1,mod.block)*sc[...,None]).reshape(q.shape).to(torch.bfloat16)
        setattr(mod,name,value)
    mod.fp8=False
    del q,sc,value
    torch.cuda.synchronize()
results['dequantize_once_seconds']=time.perf_counter()-started

def grouped(x):
    logits=torch.nn.functional.linear(x.float(),mod.gate.float())
    scores=torch.sigmoid(logits)
    select=logits-mod.qb_beta.float() if mod.quantile else scores+mod.e_score_correction_bias.float()
    idx=torch.topk(select,mod.top_k,dim=-1).indices
    weights=scores.gather(-1,idx)
    if mod.top_k>1:weights=weights/(weights.sum(-1,keepdim=True)+1e-20)
    weights=(weights*mod.scaling).to(x.dtype)
    flat=idx.flatten();order=torch.argsort(flat,stable=True)
    tok=order//mod.top_k
    counts=torch.bincount(flat,minlength=mod.gate.shape[0])
    offsets=counts.cumsum(0).to(torch.int32)
    gu=torch._grouped_mm(x[tok],mod.gate_up_proj.transpose(-1,-2),offs=offsets)
    gate,up=gu.chunk(2,-1)
    h=act(gate,up,mod.limit,mod.expert_norm)
    y=torch._grouped_mm(h,mod.down_proj.transpose(-1,-2),offs=offsets)
    out=torch.zeros_like(x)
    out.index_add_(0,tok,(y*weights.flatten()[order,None]).to(x.dtype))
    if mod.shared_gate_proj is not None:
        out=out+mod.shared_down_proj(act(mod.shared_gate_proj(x),mod.shared_up_proj(x),mod.limit,mod.shared_norm))
    return out

for n,x in xs.items():
    seconds,out=timed(mod,x)
    row={'tokens':n,'reference_seconds':times[n],'cached_seconds':seconds,
         'cached_bitwise_equal':torch.equal(out,refs[n])}
    try:
        seconds,fast=timed(grouped,x)
        diff=(fast.float()-refs[n].float())
        row.update(grouped_seconds=seconds,max_abs_diff=float(diff.abs().max()),
                   rms_diff=float(diff.square().mean().sqrt()),reference_rms=float(refs[n].float().square().mean().sqrt()))
    except Exception as e:row['grouped_error']=repr(e)
    results['trials'].append(row);print(json.dumps(row),flush=True)
Path(a.out).write_text(json.dumps(results,indent=2)+'\n')
