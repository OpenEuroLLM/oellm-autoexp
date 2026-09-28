"""Prepare and launch isolated, source-pinned B1 continuation recipes.

Preparation and GPU submission both require --execute. Existing run directories
are never overwritten; checkpoints, sources and restore contracts are recorded.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8*1024*1024): digest.update(chunk)
    return digest.hexdigest()


def git(repo, *args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()


def safe_path(value):
    path = Path(value).expanduser().absolute()
    if not re.fullmatch(r'/[A-Za-z0-9_./:+-]+',str(path)):
        raise ValueError('Paths must be absolute and contain no shell/mount metacharacters or spaces')
    return path


def write(path, value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def contract(variant, start, end, ramp=0, floor=.1, run_name=None):
    spec = read(REPO/'versions/b1_recipes.json')
    if variant not in spec['variants'] or not 0 < start < end <= 894000 or ramp < 0 or not 0 < floor <= 1:
        raise ValueError('Invalid variant, continuation range or LR ramp')
    args = dict(spec['args'],wandb_exp_name=run_name or spec['variants'][variant]['label'])
    return dict(args=args,scheduler=dict(max_lr=3e-4,min_lr=0.,lr_decay_style='WSD',
        override_opt_param_scheduler=True,use_checkpoint_opt_param_scheduler=False),
        start=start,end=end,anchor_samples=start*4096,warmup_updates=max(ramp,1),
        floor=floor if ramp else 1.,target_lr=3e-4,weight_decay=.1)


def check_source(source, revision):
    if git(source,'rev-parse','HEAD') != revision or git(source,'status','--porcelain','--untracked-files=no'):
        raise ValueError('Wrong or modified Megatron source')
    if not (source/'pretrain_gpt.py').is_file(): raise ValueError('Megatron submodule not initialized')


def prepare(a):
    from production_walltime import resolve
    spec = read(REPO/'versions/b1_recipes.json')
    variant = spec['variants'][a.variant]
    root = safe_path(a.run_root)
    if root.exists(): raise ValueError('Use a fresh run root; never overwrite a prepared run')
    checkpoint = safe_path(a.checkpoint).resolve(strict=True)
    match = re.fullmatch(r'iter_(\d{7})',checkpoint.name)
    shards = list(checkpoint.glob('*.distcp'))
    if not match or not (checkpoint/'.metadata').is_file() or len(shards)!=a.checkpoint_shards or any(p.stat().st_size<=0 for p in shards):
        raise ValueError('An explicit complete native iter_NNNNNNN checkpoint is required')
    start = int(match[1]); name=a.name or variant['label']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',name): raise ValueError('Unsafe run name')
    if a.nodes < 16 or a.nodes % 4: raise ValueError('TP4/PP4 requires node count divisible by4; production default512')
    rc=contract(a.variant,start,a.end_step,a.ramp_updates,a.ramp_floor,name)
    rc['args']['wandb_project']=a.wandb_project
    source_repo = safe_path(a.source_repository or REPO/'submodules/Megatron-LM')
    if not (source_repo/'pretrain_gpt.py').is_file(): raise ValueError('Initialize Megatron or supply --source-repository')
    git(source_repo,'cat-file','-e',variant['commit']+':pretrain_gpt.py')
    paths={k:safe_path(getattr(a,k)) for k in ('data_manifest','data_cache','tokenizer','image','exclude_file')}
    for key,path in paths.items():
        if not path.exists(): raise ValueError('Missing input '+key+': '+str(path))
    wall=resolve(a.account,a.partition,a.qos)
    plan=dict(variant=a.variant,source_commit=variant['commit'],checkpoint=str(checkpoint),start=start,end=a.end_step,
              nodes=a.nodes,walltime=wall,run_root=str(root),ramp_updates=a.ramp_updates,
              paths={k:str(v) for k,v in paths.items()},gpu_submission=False)
    if not a.execute: print(json.dumps(plan,indent=2)); return
    if sha(paths['image']) != spec['image_sha256']:
        raise ValueError('Container differs from the validated JUPITER image')
    root.mkdir(parents=True); (root/'config').mkdir()
    source=root/'Megatron-LM'
    subprocess.run(['git','clone','--no-hardlinks','--no-checkout',str(source_repo),str(source)],check=True)
    subprocess.run(['git','-C',str(source),'checkout','--detach',variant['commit']],check=True)
    check_source(source,variant['commit'])
    seeds=root/'training_ckpts/checkpoints'; seeds.mkdir(parents=True)
    (seeds/checkpoint.name).symlink_to(checkpoint,target_is_directory=True)
    (seeds/'latest_checkpointed_iteration.txt').write_text(str(start)+'\n')
    (root/'training_ckpts/checkpoints_rolling').mkdir()
    aux=dict(plan['paths'],run_root=str(root),repo_root=str(REPO),source=str(source),account=a.account,
             partition=a.partition,qos=a.qos,walltime=wall['time'],end_step=a.end_step,remaining_steps=a.end_step-start,nodes=a.nodes,
             wandb_project=a.wandb_project,run_name=name)
    # The recipe names, rather than the inherited revival experiment, select B1.
    cfg=dict(hydra={'searchpath':['file://'+str(REPO/'config')]},defaults=[f'/experiments/oellm_32b_dense/b1_reborn_{a.variant}','_self_'],aux=aux)
    write(root/'config/b1.yaml',cfg); write(root/'restore-contract.json',rc)
    files=[REPO/'versions/b1_recipes.json',root/'config/b1.yaml',root/'restore-contract.json',checkpoint/'.metadata',paths['data_manifest']]
    files += [REPO/'tools'/n for n in ('b1_run.py','conservative_b1_training.py','gradient_transition_warmup.py','production_walltime.py')]
    files += sorted((REPO/'config').rglob('*.yaml'))
    files += sorted((REPO/'templates').glob('*.sbatch'))
    plan.update(source=str(source),source_tree=git(source,'rev-parse','HEAD^{tree}'),autoexp=git(REPO,'rev-parse','HEAD'),
                image_sha256=spec['image_sha256'],checkpoint_shards=len(shards),pins={str(p):sha(p) for p in files})
    write(root/'b1-run.json',plan)
    command=[sys.executable,str(REPO/'tools/b1_run.py'),'launch','--run-root',str(root),'--execute']
    (root/'run.sh').write_text('#!/bin/bash\nset -euo pipefail\nexec '+shlex.join(command)+'\n')
    print(json.dumps(plan,indent=2))


def verify(root):
    m=read(root/'b1-run.json')
    if git(REPO,'rev-parse','HEAD')!=m['autoexp'] or git(REPO,'status','--porcelain','--untracked-files=no'):
        raise ValueError('Autoexp release changed; use the prepared clean checkout')
    for path,digest in m['pins'].items():
        if sha(path)!=digest: raise ValueError('Prepared input changed: '+path)
    check_source(Path(m['source']),m['source_commit'])
    if (root/'STOP').exists(): raise ValueError('Operator STOP')
    return m


def launch(a):
    root=safe_path(a.run_root); m=verify(root)
    from production_walltime import resolve
    wall=resolve(m['walltime']['account'],m['walltime']['partition'],m['walltime']['qos'])
    if wall['time']!=m['walltime']['time']: raise ValueError('Scheduler policy changed; review and re-prepare before launch')
    command=[sys.executable,str(REPO/'scripts/run_autoexp.py'),'--config-dir',str(root/'config'),
             '--config-name','b1','--monitor-state-dir',str(root/'monitor_state')]
    if not a.execute: command.append('--dry-run')
    env=dict(os.environ,HYDRA_STAGED_SWEEP_CACHE='0',PYTHONPATH=str(REPO),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',
             JUPITER_EXCLUDE_NODES=m['paths']['exclude_file'])
    with (root/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.execute:
            # A lost monitor must be reattached, never create a second training run.
            with (root/'launch-intent.json').open('x') as intent:
                json.dump(dict(command=command,autoexp=m['autoexp']),intent)
        subprocess.run(command,cwd=REPO,env=env,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('prepare'); q.add_argument('--variant',choices=['legacy','patched'],required=True)
    q.add_argument('--run-root',required=True);q.add_argument('--checkpoint',required=True)
    for name in ('data-manifest','data-cache','tokenizer','image','exclude-file','account'): q.add_argument('--'+name,required=True)
    q.add_argument('--checkpoint-shards',type=int,default=2048);q.add_argument('--source-repository');q.add_argument('--name');q.add_argument('--partition',default='booster')
    q.add_argument('--qos',default='normal');q.add_argument('--nodes',type=int,default=512)
    q.add_argument('--end-step',type=int,default=894000);q.add_argument('--wandb-project',default='oellm_32B_dense')
    q.add_argument('--ramp-updates',type=int,default=0);q.add_argument('--ramp-floor',type=float,default=.1)
    q.add_argument('--execute',action='store_true')
    q=sub.add_parser('launch');q.add_argument('--run-root',required=True);q.add_argument('--execute',action='store_true')
    q=sub.add_parser('verify');q.add_argument('--run-root',required=True)
    a=p.parse_args()
    if a.action=='prepare': prepare(a)
    elif a.action=='launch': launch(a)
    else: print(json.dumps(verify(safe_path(a.run_root)),indent=2))


if __name__ == '__main__': main()
