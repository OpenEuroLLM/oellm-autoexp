"""Prepare/check exact phase-shaped caches using the production prebuild routine."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
FIELDS=('data_args_path','data_cache_path','tokenizer_type','tokenizer_model','seq_length','seed','split',
        'global_batch_size','eval_global_batch_size','eval_micro_batch_size','eval_iters','eval_interval',
        'start_eval_at_iter','mid_level_dataset_surplus','dataloader_inter_document_masking',
        'reset_attention_mask','reset_position_ids','eod_mask_loss','create_attention_mask_in_dataloader')

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tokenizer_identity(path):
    path=Path(path)
    if path.is_file():return {str(path):sha(path)}
    files=sorted(p for p in path.iterdir() if p.is_file())
    if not files:raise ValueError('Tokenizer directory is empty')
    return {str(p):sha(p) for p in files}
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)

def shape(m,iteration):
    g=m['global_batch_size'];end=m['train_samples']//g
    points=end//m['eval_interval']+1-(m.get('start_eval_at_iter') or 0)//m['eval_interval']
    evaluation=m['eval_iters']*(m.get('eval_global_batch_size') or g)
    transitions=[0]+[int(s) for s in str(m.get('phase_transition_iterations') or '').split(',') if s]+[end]
    n=min(s for s in transitions if s>iteration)-max(s for s in transitions if s<=iteration)
    return n*g,max(0,points)*evaluation,evaluation

def normalized(m,iteration):
    if int(m['split'].replace('/',',').split(',')[-1])!=0:raise ValueError('Prebuild adapter requires zero test split')
    train,valid,test=shape(m,iteration)
    if valid<=0:raise ValueError('Use positive validation sampling; eval_iters=0 creates an invalid empty sampler')
    eg=m.get('eval_global_batch_size') or m['global_batch_size']
    data={k:m[k] for k in FIELDS if k in m}
    data.update(train_samples=train,train_iters=None,phase_transition_iterations=None,
        eval_iters=valid//eg,eval_interval=train//m['global_batch_size']+1,start_eval_at_iter=None,
        lr_warmup_samples=0,lr_warmup_iters=0,lr_decay_iters=None,lr_wsd_decay_iters=None,
        lr_decay_samples=train,lr_wsd_decay_samples=train)
    return data

def resolved(root):
    sys.path.insert(0,str(REPO))
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    from oellm_autoexp.config.loader import ensure_registrations
    ensure_registrations()
    with initialize_config_dir(config_dir=str(root/'config'),version_base=None):
        return OmegaConf.to_container(compose(config_name='annealing_arm'),resolve=True)

def prepare(root):
    from omegaconf import OmegaConf
    if (root/'STOP').exists():raise ValueError('Preserve STOP')
    target=resolved(root);recipe=read(root/'recipe.json');m=target['backend']['megatron']
    data=normalized(m,recipe['seed_iteration']);folder=root/'cache-preparation'
    folder.mkdir(exist_ok=False)
    cfg=dict(defaults=['/experiments/oellm_32b_dense/data_cache_build_fscratch_trainshape','_self_'],
        job=dict(name='cache-'+root.name,base_output_dir=str(folder),log_path=str(folder/'slurm-%j.log'),
            log_path_current=str(folder/'current.log'),config_path=str(folder/'config-%j.yaml'),
            config_path_current=str(folder/'current.yaml'),est_steps=1,chain_repeat=1),
        backend=dict(megatron=data,torchrun_args=dict(nnodes=1,nproc_per_node=1)),
        container=dict(image=target['container']['image'],python=target['container']['python']),
        slurm=dict(sbatch=dict(account=target['slurm']['sbatch']['account'],time='04:00:00',nodes=1),
                   env=dict(HF_HOME=str(folder/'hf-cache'))))
    OmegaConf.save(cfg,root/'config/cache_preparation.yaml')
    command=[sys.executable,str(REPO/'scripts/run_autoexp.py'),'--config-dir',str(root/'config'),
        '--config-name','cache_preparation','--monitor-state-dir',str(folder/'monitor'),'--no-monitor','--dry-run']
    subprocess.run(command,cwd=REPO,env=dict(os.environ,PYTHONPATH=str(REPO)),check=True)
    save(folder/'settings.json',dict(target=m,normalized=data,samples=shape(m,recipe['seed_iteration']),
        target_config_sha256=sha(root/'config/annealing_arm.yaml'),datamix_sha256=sha(m['data_args_path']),
        tokenizer_sha256=tokenizer_identity(m['tokenizer_model']),cache_path=m['data_cache_path'],
        maximum_gpu_hours=16,command=command,source=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()))
    # One writer across preparation and later training. The lock covers the allocation.
    script=next(folder.glob('job_*.sbatch'));text=script.read_text();needle='echo "Launching '
    i=text.index(needle)
    text=text[:i]+'exec 9>'+shlex.quote(str(Path(m['data_cache_path'])/'.writer.lock'))+'\nflock -n 9 || exit 75\n'+text[i:]
    script.write_text(text);subprocess.run(['bash','-n',str(script)],check=True)
    save(folder/'render.json',dict(script=str(script),script_sha256=sha(script),time=time.time()))
    print('Prepared cache allocation:',script)

def submit(root):
    folder=root/'cache-preparation';render=read(folder/'render.json')
    if (root/'STOP').exists():raise ValueError('Preserve STOP')
    if sha(render['script'])!=render['script_sha256']:raise ValueError('Prebuild script changed')
    intent=folder/'submission-intent.json'
    if intent.exists() or (folder/'submitted.json').exists() or (folder/'reuse.json').exists():
        raise ValueError('Existing preparation or submission intent: reconcile it before submitting')
    comment='anneal-cache-'+hashlib.sha256(str(root).encode()).hexdigest()[:16]
    command=['sbatch','--parsable','--comment='+comment,render['script']]
    with intent.open('x') as stream:json.dump(dict(command=command,time=time.time(),comment=comment),stream)
    result=subprocess.run(command,cwd=REPO,text=True,capture_output=True)
    job=result.stdout.strip().split(';')[0]
    if result.returncode or not job.isdigit():raise ValueError('Uncertain submission; intent retained: '+result.stderr)
    save(folder/'submitted.json',dict(job=job,script=render['script'],script_sha256=render['script_sha256'],
        time=time.time(),maximum_gpu_hours=16,comment=comment))
    (folder/'submitted.txt').write_text(job+'\n');print('Submitted prebuild',job)

def coverage(root,settings):
    m=settings['target'];cache=Path(m['data_cache_path']);fields=shlex.split(Path(m['data_args_path']).read_text())
    weights=list(map(float,fields[::2]));weights=[w/sum(weights) for w in weights];prefixes=fields[1::2]
    evidence={}
    for split,number in zip(('train','valid'),settings['samples'][:2]):
        size=sum(math.ceil(number*w) for w in weights);matches=[]
        for p in cache.glob('*-BlendedDataset-'+split+'-description.txt'):
            d=read(p)
            if d.get('size')!=size or [c['dataset_path'] for c in d['datasets']]!=prefixes:continue
            if any(abs(a-b)>1e-12 for a,b in zip(d['weights'],weights)):continue
            if len(d['datasets'])!=len(weights):continue
            valid=True
            for child,w in zip(d['datasets'],weights):
                expected=int(math.ceil(math.ceil(number*w)*(1+m['mid_level_dataset_surplus'])))
                if child['num_samples']!=expected or child['random_seed']!=m['seed'] or child['sequence_length']!=m['seq_length'] or child['split']!=m['split']:valid=False;break
                if child['tokenizer']['tokenizer_path']!=m['tokenizer_model']:valid=False;break
                content=json.dumps(child,indent=4);digest=hashlib.md5(content.encode(),usedforsecurity=False).hexdigest()
                child_path=cache/(digest+'-GPTDataset-'+split+'-description.txt')
                if not child_path.exists() or child_path.read_text()!=content:valid=False;break
            if valid:matches.append(p)
        if len(matches)!=1:raise ValueError(f'Expected one complete matching {split} blend, found {len(matches)}')
        evidence[split]=dict(requested_samples=number,blend_samples=size,prefixes=len(prefixes),description=str(matches[0]),sha256=sha(matches[0]))
    return evidence

def reuse(root, prior):
    """Bind a fresh arm to a successful preparation with identical cache inputs."""
    folder = root / 'cache-preparation'
    old = prior / 'cache-preparation'
    if any((folder / name).exists() for name in ('submitted.json', 'submission-intent.json', 'reuse.json')):
        raise ValueError('Preparation already bound or submitted')
    if (old / 'reuse.json').exists():
        raise ValueError('Use the original preparation allocation for reuse')
    current, previous = (read(d / 'settings.json') for d in (folder, old))
    for key in ('samples', 'datamix_sha256', 'tokenizer_sha256', 'cache_path'):
        if current[key] != previous[key]:
            raise ValueError('Incompatible cache reuse: ' + key)
    new_data, old_data = (dict(d['normalized']) for d in (current, previous))
    old_data['data_args_path'] = new_data['data_args_path']
    if new_data != old_data:
        raise ValueError('Incompatible normalized cache settings')
    gitlink = lambda source: subprocess.check_output(
        ['git', '-C', str(REPO), 'ls-tree', source, '--', 'submodules/Megatron-LM'], text=True)
    if gitlink(current['source']) != gitlink(previous['source']):
        raise ValueError('Cache preparation used a different Megatron revision')
    previous_proof = check(prior, write_receipt=False)
    submission = read(old / 'submitted.json')
    script = Path(read(old / 'render.json')['script'])
    log = old / ('slurm-' + submission['job'] + '.log')
    pins = {str(path): sha(path) for path in (script, log, old / 'settings.json', old / 'submitted.json')}
    save(folder / 'reuse.json', dict(prior=str(prior), job=submission['job'], log=str(log),
                                     pins=pins, coverage=previous_proof))
    save(folder / 'submitted.json', submission)
    check(root)


def check(root, write_receipt=True):
    folder=root/'cache-preparation';settings=read(folder/'settings.json');render=read(folder/'render.json')
    if (root/'STOP').exists():raise ValueError('Preserve STOP')
    if settings['target_config_sha256']!=sha(root/'config/annealing_arm.yaml'):raise ValueError('Training configuration changed')
    m=settings['target']
    if sha(m['data_args_path'])!=settings['datamix_sha256'] or tokenizer_identity(m['tokenizer_model'])!=settings['tokenizer_sha256']:raise ValueError('Data/tokenizer identity changed')
    if sha(render['script'])!=render['script_sha256']:raise ValueError('Prebuild script changed')
    submitted=read(folder/'submitted.json');job=submitted['job']
    if not re.fullmatch(r'\d+',job):raise ValueError('Invalid job')
    result=subprocess.check_output(['sacct','-X','-nP','-j',job,'--format=JobIDRaw,State'],text=True)
    states=[s.split('|')[1] for s in result.splitlines() if s.split('|')[0]==job]
    if not states or any(s!='COMPLETED' for s in states):raise ValueError('Prebuild not completed: '+str(states))
    reuse_path = folder / 'reuse.json'
    if reuse_path.exists():
        provenance = read(reuse_path)
        if provenance['job'] != job or any(sha(path) != digest for path, digest in provenance['pins'].items()):
            raise ValueError('Prior cache preparation evidence changed')
        log = Path(provenance['log'])
    else:
        if submitted['script_sha256'] != render['script_sha256']:
            raise ValueError('Preparation allocation used a different script')
        log=folder/('slurm-'+job+'.log')
    if not re.search(r'exiting program at iteration\s+1\b',log.read_text()):raise ValueError('No successful one-update completion evidence')
    cmd=[sys.executable,str(REPO/'scripts/check_data_cache.py'),m['data_cache_path']]
    result=subprocess.run(cmd,text=True,capture_output=True)
    if write_receipt:
        (folder/'check.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise ValueError('Production cache checker failed; see check.log')
    proof=coverage(root,settings)
    if reuse_path.exists() and proof != provenance['coverage']:
        raise ValueError('Reused cache coverage evidence changed')
    if write_receipt:
        save(folder/'ready.json',dict(job=job,log_sha256=sha(log),settings_sha256=sha(folder/'settings.json'),coverage=proof,time=time.time()))
    print(json.dumps(proof,indent=2))
    return proof

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','submit','check','reuse'])
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--from-root',type=Path)
    args=p.parse_args()
    if args.action == 'reuse':
        if not args.from_root:
            p.error('reuse requires --from-root')
        reuse(args.root.resolve(), args.from_root.resolve())
    else:
        globals()[args.action](args.root.resolve())
