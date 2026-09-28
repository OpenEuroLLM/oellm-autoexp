"""Historical evidence replay and real restart lifecycle with fake scheduler I/O."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from oellm_autoexp.monitor import nccl_peers as peers
from oellm_autoexp.monitor.actions import ActionContext, EventRecord, RestartAction, RestartActionConfig
from oellm_autoexp.monitor.loop import JobRecord, JobRuntime, MonitorLoop
from oellm_autoexp.slurm_gen.generator import build_sbatch_directives
from oellm_autoexp.slurm_gen.schema import SlurmConfig, SbatchConfig

EVIDENCE = json.loads((Path(__file__).parent/'fixtures/nccl_peer_1725379.json').read_text())


def setup(tmp_path):
    e = EVIDENCE
    job = dict(job_id=e['job_id'], state=e['state'], start=peers.stamp(e['start']),
               end=peers.stamp(e['end']), hosts=[e['reporter'], e['peer']])
    nodes = {e['peer']: copy.deepcopy(e['node']),
             e['reporter']: dict(NodeName=e['reporter'], NodeAddr='10.0.0.1', State='IDLE')}
    (tmp_path/f"nccl-{e['reporter']}.log").write_text(e['warning']+'\n')
    return job, nodes


def test_exact_historical_warning_excludes_peer_not_reporter(tmp_path):
    job, nodes = setup(tmp_path)
    result = peers.scan(tmp_path, job, nodes)
    assert result['exclude'] == ['jpbo-071-42']
    exclusions=tmp_path/'exclude'; exclusions.write_text('# preserve\njpbo-001-01\n')
    audit=tmp_path/'events.jsonl'
    assert peers.apply(result, exclusions, audit) == ['jpbo-071-42']
    assert peers.apply(result, exclusions, audit) == []
    assert exclusions.read_text().startswith('# preserve\njpbo-001-01\n')
    assert exclusions.read_text().splitlines().count('jpbo-071-42') == 1
    assert 'jpbo-022-09' not in exclusions.read_text()
    assert json.loads(audit.read_text().splitlines()[0])['events'][0]['log_line'] == EVIDENCE['warning']


@pytest.mark.parametrize('case', ['healthy','down_only','unresponsive_only','old_busy','missing_busy',
                                  'foreign_peer','ambiguous_ip','old_warning','late_warning',
                                  'wrong_reporter','completed','running','cuda_cleanup'])
def test_insufficient_or_stale_evidence_never_excludes(tmp_path, case):
    job,nodes=setup(tmp_path); node=nodes[EVIDENCE['peer']]
    if case=='healthy': node['State']='IDLE'
    if case=='down_only': node['State']='DOWN'
    if case=='unresponsive_only': node['State']='ALLOCATED+NOT_RESPONDING'
    if case=='old_busy': node['LastBusyTime']='2026-09-08T10:40:21'
    if case=='missing_busy': node.pop('LastBusyTime')
    if case=='foreign_peer': job['hosts'].remove(EVIDENCE['peer'])
    if case=='ambiguous_ip': nodes[EVIDENCE['reporter']]['NodeAddr']=node['NodeAddr']
    if case=='old_warning': job['start']=job['end']+100
    if case=='late_warning': job['end']-=200
    if case=='completed': job['state']='COMPLETED'
    if case=='running': job['state']='RUNNING'
    p=tmp_path/f"nccl-{EVIDENCE['reporter']}.log"
    if case=='wrong_reporter': p.write_text(EVIDENCE['warning'].replace('jpbo-022-09','jpbo-090-25'))
    if case=='cuda_cleanup': p.write_text("[2026-09-09 19:17:41] jpbo-090-25:366286:375408 [1] misc/strongstream.cc:403 NCCL WARN Cuda failure 'unspecified launch failure'\n")
    assert peers.scan(tmp_path,job,nodes)['exclude']==[]


def test_mass_failure_requires_review(tmp_path):
    job,nodes=setup(tmp_path)
    result=peers.scan(tmp_path,job,nodes,cap=0)
    assert result['exclude']==[] and result['refused']


def test_missing_exclusion_file_or_audit_failure_does_not_create_exclusions(tmp_path):
    job,nodes=setup(tmp_path);result=peers.scan(tmp_path,job,nodes)
    f=tmp_path/'missing'
    with pytest.raises(ValueError): peers.apply(result,f,tmp_path/'audit')
    f.write_text('jpbo-001-01\n')
    with pytest.raises(OSError): peers.apply(result,f,tmp_path/'absent'/'audit')
    assert f.read_text()=='jpbo-001-01\n'


def fake_scheduler(tmp_path, monkeypatch):
    e=EVIDENCE;bin_dir=tmp_path/'bin';bin_dir.mkdir()
    responses={'sacct':f"{e['job_id']}|FAILED|{e['start']}|{e['end']}|{e['reporter']},{e['peer']}|\n",
               'hosts':e['reporter']+'\n'+e['peer']+'\n',
               'nodes':' '.join(f'{k}={v}' for k,v in e['node'].items())+'\n'+f"NodeName={e['reporter']} NodeAddr=10.0.0.1 State=IDLE\n"}
    for name in ['sacct','scontrol']:
        p=bin_dir/name
        p.write_text(f'#!{sys.executable}\nimport sys\ndata={responses!r}\nprint(data["sacct" if {name!r}=="sacct" else "hosts" if "hostnames" in sys.argv else "nodes"],end="")\n')
        p.chmod(0o755)
    import os
    monkeypatch.setenv('PATH',str(bin_dir)+os.pathsep+os.environ['PATH'])


@pytest.mark.parametrize('fault',[None,'timeout','error'])
def test_cancel_teardown_real_peer_hook_then_render_excludes(tmp_path,monkeypatch,fault):
    setup(tmp_path);fake_scheduler(tmp_path,monkeypatch)
    exclude=tmp_path/'exclude';exclude.write_text('jpbo-001-01\n')
    logs=tmp_path/'logs';logs.mkdir();log=logs/f"slurm-{EVIDENCE['job_id']}.log";log.touch()
    slurm=SlurmConfig(name='fake',template_path=str(tmp_path/'unused'),script_dir=str(tmp_path),
                      log_dir=str(logs),exclude_file=str(exclude),sbatch=SbatchConfig())
    definition=SimpleNamespace(slurm=slurm)
    job=JobRecord(job_id='fake',definition=definition,runtime=JobRuntime(submitted=True,
                  runtime_job_id=EVIDENCE['job_id'],last_status='RUNNING',attempts=1))
    loop=MonitorLoop.__new__(MonitorLoop);loop._pending_restart_hooks={}
    actions=[]
    loop._store=SimpleNamespace(upsert=lambda j:None)
    loop._get_client=lambda j:SimpleNamespace(cancel=lambda rid:actions.append('cancel'),remove=lambda rid:None)
    loop._build_job_metadata=lambda j:{}
    loop._resolve_log_path=lambda j:log
    loop._restart_job=lambda j:actions.append('\n'.join(build_sbatch_directives(slurm)))
    cfg=RestartActionConfig(cancel_first=True,nccl_peer_scan=True,exclude_file=str(exclude))
    res=RestartAction(cfg).execute(ActionContext(event=EventRecord(event_id='x',name='x',source='log'),job_metadata={}))
    assert res.metadata['nccl_peer_scan'] is True
    loop._pending_restart_hooks[job.job_id]=res.metadata
    assert loop._apply_effect(job,'restart',EVIDENCE['job_id'])
    assert actions==['cancel'] and not (tmp_path/'nccl-peer-events.jsonl').exists()
    assert loop._resume_deferred_restart(job,EVIDENCE['job_id'])
    assert actions==['cancel']
    job.runtime.last_status='CANCELLED'
    if fault:
        def fail(*args,**kwargs):
            if fault=='timeout': raise subprocess.TimeoutExpired('scan',60)
            raise OSError('scanner unavailable')
        monkeypatch.setattr(subprocess,'run',fail)
    assert loop._resume_deferred_restart(job,EVIDENCE['job_id'])
    assert len(actions)==2 and actions[0]=='cancel'
    assert ('jpbo-071-42' in actions[1]) == (fault is None)
    assert 'jpbo-022-09' not in actions[1]
    assert job.runtime.deferred_restart=={}


def test_size_limit_fails_without_partial_apply(tmp_path):
    job,nodes=setup(tmp_path)
    with (tmp_path/f"nccl-{EVIDENCE['peer']}.log").open('wb') as f: f.truncate(17*1024**2)
    with pytest.raises(ValueError,match='size limit'): peers.scan(tmp_path,job,nodes)


def test_busy_exclusion_lock_times_out_without_writing(tmp_path, monkeypatch):
    import fcntl
    job,nodes=setup(tmp_path);result=peers.scan(tmp_path,job,nodes)
    exclusions=tmp_path/'exclude';exclusions.write_text('jpbo-001-01\n')
    with exclusions.open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX)
        ticks=iter([0,4]);monkeypatch.setattr(peers.time,'monotonic',lambda:next(ticks))
        with pytest.raises(TimeoutError): peers.apply(result,exclusions,tmp_path/'audit')
    assert exclusions.read_text()=='jpbo-001-01\n'
    assert not (tmp_path/'audit').exists()


def test_snapshot_refuses_missing_accounting_without_exclusion(monkeypatch):
    monkeypatch.setattr(peers,'command',lambda argv:'')
    with pytest.raises(ValueError,match='allocation'): peers.snapshot('1725379')
    with pytest.raises(ValueError,match='numeric'): peers.snapshot('1725379; echo bad')
