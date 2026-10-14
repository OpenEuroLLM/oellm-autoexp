"""Explicit experiment-only Adam reset and post-resume LR warmup.

Retains model/master weights, RNG, data position, weight-decay policy and global
training iteration. Full moment reset also resets Adam bias-correction age.
No head-only reset: this Megatron version checkpoints a common optimizer age.
"""
import json
import math
import os
from pathlib import Path
import runpy
import sys
import types


def warmup_factor(completed_updates, warmup_updates, floor=.1):
    if completed_updates<0 or warmup_updates<0 or not 0<floor<=1:
        raise ValueError('Invalid transition warmup')
    if not warmup_updates:return 1.
    return floor+(1-floor)*min(completed_updates/warmup_updates,1.)


def leaves(optimizer):
    children=getattr(optimizer,'chained_optimizers',None)
    if children is not None:
        for child in children:yield from leaves(child)
    elif getattr(optimizer,'is_stub_optimizer',False):return
    elif hasattr(optimizer,'optimizer'):yield from leaves(optimizer.optimizer)
    elif hasattr(optimizer,'state') and hasattr(optimizer,'param_groups'):yield optimizer
    else:raise TypeError('Unsupported optimizer wrapper')


def reset_adam(optimizer):
    """Validate all local shards before mutation; leave every non-moment tensor intact."""
    targets=[];ages=[];groups=0
    for leaf in leaves(optimizer):
        if type(leaf).__name__ not in ('Adam','AdamW','FusedAdam'):
            raise TypeError('Only validated Adam implementations are supported')
        for group in leaf.param_groups:
            if group.get('amsgrad',False):raise ValueError('AMSGrad not supported')
            if 'step' in group:ages.append((group,'step'))
            groups+=1
            for p in group['params']:
                state=leaf.state.get(p,{})
                if not state or 'exp_avg' not in state or 'exp_avg_sq' not in state:
                    raise ValueError('Expected fully restored Adam moments')
                for key in ('exp_avg','exp_avg_sq'):
                    tensor=state[key]
                    if tensor.shape!=p.shape or not tensor.is_floating_point():
                        raise ValueError('Unexpected Adam moment layout')
                    # Current production uses unscaled FP32 moments. Quantized
                    # optimizer state needs its own explicit reset implementation.
                    if str(tensor.dtype)!='torch.float32':raise ValueError('Expected FP32 Adam moments')
                    targets.append(tensor)
                if 'step' in state:ages.append((state,'step'))
    if targets and not ages:raise ValueError('Adam bias-correction age not found')
    prior=sorted({int(holder[key].item() if hasattr(holder[key],'item') else holder[key]) for holder,key in ages})
    if len(prior)>1:raise ValueError('Optimizer ages differ before reset')
    for tensor in targets:tensor.zero_()
    for holder,key in ages:
        if hasattr(holder[key],'zero_'):holder[key].zero_()
        else:holder[key]=0
    return dict(moment_tensors=len(targets),moment_elements=sum(t.numel() for t in targets),
                optimizer_groups=groups,previous_ages=prior,new_age=0)


def attach_warmup(scheduler, batch_size, updates, floor, anchor_samples=None):
    if batch_size<=0:raise ValueError('Positive global batch size required')
    start=scheduler.num_steps if anchor_samples is None else anchor_samples
    if start < 0 or scheduler.num_steps < start:
        raise ValueError('Scheduler precedes transition anchor')
    original=scheduler.get_lr
    def get_lr(self,group):
        elapsed=(self.num_steps-start)/batch_size
        return original(group)*warmup_factor(elapsed,updates,floor)
    scheduler.get_lr=types.MethodType(get_lr,scheduler)
    scheduler.step(increment=0)
    return dict(start_scheduler_samples=start,batch_size=batch_size,warmup_updates=updates,floor=floor)


def transition_action(iteration, start, end, mode, resumable):
    """Reset exactly at the original fork, never at a recovered checkpoint."""
    if not start <= iteration < end or (not resumable and iteration != start):
        raise ValueError('Wrong transition checkpoint')
    return mode == 'reset_all' and iteration == start


def adam_ages(optimizer):
    values=set()
    for leaf in leaves(optimizer):
        for group in leaf.param_groups:
            if 'step' in group:
                value=group['step'];values.add(int(value.item() if hasattr(value,'item') else value))
        for state in leaf.state.values():
            if 'step' in state:
                value=state['step'];values.add(int(value.item() if hasattr(value,'item') else value))
    return sorted(values)


def main():
    source=Path(os.environ['TRANSITION_SOURCE']).resolve()
    output=Path(os.environ['TRANSITION_OUT']).resolve()
    expected=int(os.environ.get('TRANSITION_START','156000'))
    resumable=os.environ.get('TRANSITION_RESUMABLE')=='1'
    end=int(os.environ.get('TRANSITION_END',str(expected+100)))
    mode=os.environ['TRANSITION_MODE']
    updates=int(os.environ.get('TRANSITION_WARMUP_UPDATES','0'))
    floor=float(os.environ.get('TRANSITION_WARMUP_FLOOR','.1'))
    if mode not in ('retain','reset_all'):raise ValueError('Unknown reset mode')
    warmup_factor(0,updates,floor)
    sys.path.insert(0,str(source))
    import megatron.training.training as training
    import torch
    original=training.setup_model_and_optimizer
    called=False
    def setup(*args,**kwargs):
        nonlocal called
        if called:raise ValueError('Transition setup executed twice')
        result=original(*args,**kwargs)
        model,optimizer,scheduler=result
        settings=training.get_args()
        do_reset=transition_action(settings.iteration,expected,end,mode,resumable)
        if (settings.save and not resumable) or settings.no_load_optim or settings.no_load_rng:
            raise ValueError('Pilot requires full restore and disabled checkpoint saving')
        if resumable:
            anchor=int(os.environ['TRANSITION_ANCHOR_SAMPLES'])
            if settings.consumed_train_samples != anchor+(settings.iteration-expected)*settings.global_batch_size:
                raise ValueError('Restored data coordinate differs from transition trajectory')
            if scheduler.num_steps != settings.consumed_train_samples:
                raise ValueError('Restored scheduler/data coordinates differ')
        else:anchor=None
        # No reset of consumed samples, model/master weights, data RNG, decay,
        # FP8 scale histories or global training iteration.
        proof=dict(iteration=settings.iteration,consumed_train_samples=settings.consumed_train_samples,
                   mode=mode,rank=torch.distributed.get_rank())
        proof['reset']=reset_adam(optimizer) if do_reset else None
        proof['optimizer_ages']=adam_ages(optimizer)
        if resumable:
            expected_age=settings.iteration-expected if mode=='reset_all' else settings.iteration
            if proof['optimizer_ages'] != [expected_age]:
                raise ValueError('Restored Adam age does not match this transition arm')
        proof['warmup']=attach_warmup(scheduler,settings.global_batch_size,updates,floor,anchor)
        proof['resumed']=settings.iteration>expected
        proof['initial_lrs']=[float(g['lr']) for g in optimizer.param_groups]
        receipt_dir=output/os.environ['SLURM_JOB_ID'] if resumable else output
        receipt_dir.mkdir(parents=True,exist_ok=True)
        with (receipt_dir/f'rank-{proof["rank"]}.json').open('x') as f:json.dump(proof,f,indent=2)
        torch.distributed.barrier()
        called=True
        return result
    training.setup_model_and_optimizer=setup
    runpy.run_path(str(source/'pretrain_gpt.py'),run_name='__main__')


if __name__=='__main__':main()
