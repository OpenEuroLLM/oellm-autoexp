"""Fixed-prompt HF prefill, decode and numerical parity for the real checkpoint."""
import argparse
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoConfig
from oellm_autoexp.hf_export.score_on_flag import layer_device_map,load_engram_tables_cpu

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--case',choices=['reference','fast'],required=True)
p.add_argument('--prompts',required=True);p.add_argument('--out',required=True)
a=p.parse_args()
os.environ['OELLM_HF_MOE']='grouped' if a.case=='fast' else 'reference'
os.environ['OELLM_MHC_IMPL']='triton' if a.case=='fast' else 'reference'
os.environ['OELLM_HF_ENGRAM_LOOKUP']='sorted' if a.case=='fast' else 'reference'
prompts=json.loads(Path(a.prompts).read_text())
started=time.perf_counter()
cfg=AutoConfig.from_pretrained(a.model,trust_remote_code=True)
dmap=layer_device_map(cfg,54,True)
model=load_engram_tables_cpu(Path(a.model),dmap).eval()
load_seconds=time.perf_counter()-started
head_device=model.lm_head.weight.device
first_device=model.model.embed_tokens.weight.device

def prefill(ids):
    ids=torch.tensor([ids],device=first_device)
    hidden=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,:-1].to(head_device)
    logits=model.lm_head(hidden).float()
    target=ids[0,1:].to(head_device)
    return logits.log_softmax(-1).gather(-1,target[:,None])[:,0].cpu().tolist()

trials=[]
with torch.no_grad():
    for trial in range(2):
        start=time.perf_counter();rows=[prefill(ids) for ids in prompts]
        for device in range(torch.cuda.device_count()):torch.cuda.synchronize(device)
        seconds=time.perf_counter()-start
        trials.append({'seconds':seconds,'prompt_logprobs':rows})
        print(json.dumps({'case':a.case,'trial':trial,'prefill_seconds':seconds,
                          'input_tokens_per_second':sum(map(len,prompts))/seconds}),flush=True)
    generated=[];start=time.perf_counter()
    for ids in prompts[:4]:
        tokens=torch.tensor([ids],device=first_device)
        output=model.generate(input_ids=tokens,max_new_tokens=8,do_sample=False,use_cache=True)
        generated.append(output[0,len(ids):].cpu().tolist())
    for device in range(torch.cuda.device_count()):torch.cuda.synchronize(device)
    decode_seconds=time.perf_counter()-start
result={'case':a.case,'load_seconds':load_seconds,'device_map':dmap,'trials':trials,
        'generated':generated,'generation_seconds_including_prefill':decode_seconds}
Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
