#!/usr/bin/env python3
"""Read the original artifact hashes without changing the source cluster."""
import json
from pathlib import Path
import subprocess

code = r'''
import hashlib,json
from pathlib import Path
root=Path('/e/scratch/e-sta-openeurollm/poeppel1')
source=Path('/e/project1/e-sta-openeurollm/poeppel1/signed_kda')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
files=[]
for label,arm in [('ckda','skda-shipped-hybrid-lowrank'),('kda','kda-sig-hybrid-lowrank')]:
 p=root/'signed_kda_runs/megatron_ladder_finegrained/ckpt_decay'/f'{arm}_983M_gbs256/decay80B/iter_0076294/mp_rank_00/model_optim_rng.pt'
 files.append({'arm':label,'role':'native','source':str(p),'relative':f'native/{label}/iter_0076294/mp_rank_00/model_optim_rng.pt','bytes':p.stat().st_size,'sha256':sha(p)})
 p=root/'eval_finegrained/export'/f'{arm}_983M_gbs256_decay80B/config.json'
 files.append({'arm':label,'role':'reference_config','source':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'config':json.loads(p.read_text())})
tree=hashlib.sha256()
for directory in ['fla','lm_scaling']:
 for p in sorted((source/directory).rglob('*.py')):
  if any(x in p.parts for x in ['__pycache__','results','harvest']):continue
  tree.update((str(p.relative_to(source))+'\0'+sha(p)+'\n').encode())
print('MANIFEST_JSON='+json.dumps({'source':str(source),'python_tree_sha256':tree.hexdigest(),'files':files}))
'''
r = subprocess.run(['ssh', '-S', 'none', '-o', 'ControlMaster=no', '-o',
                    'BatchMode=yes', 'jupiter', 'python3 -'], input=code,
                   text=True, capture_output=True, check=True)
line = next(s for s in r.stdout.splitlines() if s.startswith('MANIFEST_JSON='))
manifest = json.loads(line.split('=', 1)[1])
out = Path(__file__).resolve().parents[1] / 'results/source_manifest.json'
out.write_text(json.dumps(manifest, indent=2) + '\n')
print(out)
