"""Post-restore B1 contract and retained-Adam warmup; no Megatron source edits."""
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import runpy
import sys

from gradient_transition_warmup import attach_warmup, adam_ages, leaves


def check_values(actual, expected):
    actual = dict(actual)
    # Megatron declares these store_true flags with default=None. Both None and
    # False mean full restore; a missing key or True must still fail the audit.
    for key in ('no_load_optim', 'no_load_rng'):
        if key in actual and actual[key] is None and expected.get(key) is False:
            actual[key] = False
    errors = {k: {'expected': v, 'actual': actual.get(k)} for k, v in expected.items()
              if actual.get(k) != v}
    if errors:
        raise ValueError('B1_POST_LOAD_CONTRACT_FAILED: '+json.dumps(errors, default=str))


def group_metadata(group):
    return copy.deepcopy({k: v for k, v in group.items() if k not in ('params', 'step')})


def restore_group_settings(groups, fresh):
    """Keep parameter identities and loaded Adam age; use the freshly built recipe."""
    if len(groups) != len(fresh):
        raise ValueError('Optimizer group count changed during checkpoint restore')
    changes = []
    for group, (params, desired) in zip(groups, fresh, strict=True):
        if tuple(map(id, group['params'])) != params:
            raise ValueError('Optimizer parameter membership changed on load')
        before = group_metadata(group)
        step = {'step': group['step']} if 'step' in group else {}
        tensors = group['params']
        group.clear()
        group.update(copy.deepcopy(desired), params=tensors, **step)
        changes.append({k: {'loaded': before.get(k), 'configured': desired.get(k)}
                        for k in set(before) | set(desired) if before.get(k) != desired.get(k)})
    return changes


def moment_sample_indices(numel, limit=32):
    """Exact integer positions: FP32 endpoints can round numel-1 up to numel."""
    if numel < 0 or limit < 1:
        raise ValueError('Invalid moment sample dimensions')
    count = min(limit, numel)
    if count <= 1:
        return [0] if count else []
    return [i * (numel - 1) // (count - 1) for i in range(count)]


def moment_evidence(optimizer):
    import torch
    digest = hashlib.sha256()
    count = 0
    nonzero = False
    for leaf in leaves(optimizer):
        for state in leaf.state.values():
            for key in ('exp_avg', 'exp_avg_sq'):
                t = state.get(key)
                if t is None:
                    raise ValueError('Missing retained Adam moment')
                # Bounded samples only; this is retention evidence, not a full tensor comparison.
                indices = torch.tensor(moment_sample_indices(t.numel()), dtype=torch.long, device=t.device)
                values = t.flatten()[indices].float().cpu()
                if not torch.isfinite(values).all():
                    raise ValueError('Nonfinite restored Adam samples')
                nonzero |= bool(torch.count_nonzero(values))
                digest.update(values.numpy().tobytes())
                count += t.numel()
    return dict(sample_sha256=digest.hexdigest(), moment_elements=count, nonzero_sample=nonzero)


def main():
    source = Path(os.environ['TRANSITION_SOURCE'])
    contract = json.loads(Path(os.environ['B1_RESTORE_CONTRACT']).read_text())
    sys.path.insert(0, str(source))
    import torch
    import megatron.training.training as training
    original_build = training.get_megatron_optimizer
    original_setup = training.setup_model_and_optimizer
    fresh = []

    def build(*args, **kwargs):
        optimizer = original_build(*args, **kwargs)
        for leaf in leaves(optimizer):
            fresh.extend((tuple(map(id, g['params'])), group_metadata(g)) for g in leaf.param_groups)
        return optimizer

    def audit(result):
        model, optimizer, scheduler = result
        settings = training.get_args()
        check_values(vars(settings), contract['args'])
        # Check constructed model/DDP objects as well as the argument namespace.
        from megatron.core.utils import get_model_config
        config_keys=('fp8','fp8_recipe','fp8_amax_history_len','fp8_amax_compute_algo',
                     'defer_embedding_wgrad_compute','tp_comm_overlap',
                     'cross_entropy_loss_fusion','cross_entropy_fusion_impl',
                     'output_z_loss_coeff','qk_layernorm')
        model_proofs=[]
        for chunk in model:
            model_config=get_model_config(chunk)
            actual={k:getattr(model_config,k) for k in config_keys}
            # The config uses a string-valued enum for the FP8 recipe.
            actual={k:getattr(v,'value',v) for k,v in actual.items()}
            check_values(actual,{k:contract['args'][k] for k in config_keys})
            ddp={k:getattr(chunk.ddp_config,k) for k in ('overlap_grad_reduce','overlap_param_gather')}
            check_values(ddp,{k:contract['args'][k] for k in ddp})
            model_proofs.append(dict(model=actual,ddp=ddp))
        if not contract['start'] <= settings.iteration < contract['end']:
            raise ValueError('Checkpoint outside authorized continuation')
        expected_samples = contract['anchor_samples'] + (settings.iteration-contract['start'])*settings.global_batch_size
        if settings.consumed_train_samples != expected_samples or scheduler.num_steps != expected_samples:
            raise ValueError('Restored iteration, data and scheduler positions disagree')
        ages = adam_ages(optimizer)
        if ages != [settings.iteration]:
            raise ValueError('Retained Adam age mismatch: '+str(ages))
        before = moment_evidence(optimizer)
        if os.environ.get('B1_GATE_EXPECT_MOMENTS'):
            prior = json.loads((Path(os.environ['B1_GATE_EXPECT_MOMENTS'])/f'rank-{torch.distributed.get_rank()}.json').read_text())
            if before != prior['moments'] or settings.iteration != prior['iteration']:
                raise ValueError('B1_POST_LOAD_CONTRACT_FAILED: rolling Adam restore differs from saved state')
        groups = [g for leaf in leaves(optimizer) for g in leaf.param_groups]
        changes = restore_group_settings(groups, fresh)
        after = moment_evidence(optimizer)
        if before != after:
            raise ValueError('Hyperparameter override changed retained Adam tensors')
        check_values(vars(scheduler), contract['scheduler'])
        base_lrs = [scheduler.get_lr(group) for group in groups]
        warmup = attach_warmup(scheduler, settings.global_batch_size, contract['warmup_updates'],
                               contract['floor'], contract['anchor_samples'])
        factor = contract['floor']+(1-contract['floor'])*min(
            (settings.iteration-contract['start'])/contract['warmup_updates'], 1)
        expected_lr = max(base_lrs) * factor
        group_proofs = []
        for group, base_lr in zip(groups, base_lrs, strict=True):
            if not math.isclose(group['lr'], base_lr*factor*group.get('lr_mult', 1), rel_tol=1e-10):
                raise ValueError('B1_POST_LOAD_CONTRACT_FAILED: effective group LR')
            if not math.isclose(group['weight_decay'], contract['weight_decay']*group.get('wd_mult', 1), abs_tol=1e-12):
                raise ValueError('B1_POST_LOAD_CONTRACT_FAILED: effective group decay')
            group_proofs.append({k: v for k, v in group_metadata(group).items() if k != 'default_config'})
        rank = torch.distributed.get_rank()
        out = Path(os.environ['TRANSITION_OUT'])/os.environ['SLURM_JOB_ID']
        out.mkdir(parents=True, exist_ok=True)
        receipt = dict(passed=True, rank=rank, iteration=settings.iteration,
                       consumed_train_samples=expected_samples, optimizer_ages=ages,
                       retained_moments=after, warmup=warmup, initial_lr=expected_lr,
                       args={k: getattr(settings, k) for k in contract['args']},
                       constructed_models=model_proofs,
                       scheduler={k: getattr(scheduler, k) for k in contract['scheduler']},
                       group_settings_reapplied=changes, effective_groups=group_proofs)
        (out/f'rank-{rank}.json').write_text(json.dumps(receipt, default=str, indent=2)+'\n')
        torch.distributed.barrier()
        if rank == 0:
            print('B1_POST_LOAD_CONTRACT_PASSED '+json.dumps(dict(iteration=settings.iteration, lr=expected_lr,
                  adam_age=ages, ranks=torch.distributed.get_world_size())), flush=True)
        return result

    def setup(*args, **kwargs):
        result=original_setup(*args, **kwargs)
        try:
            return audit(result)
        except Exception:
            print('B1_POST_LOAD_CONTRACT_FAILED: restore audit aborted; inspect traceback',flush=True)
            raise

    training.get_megatron_optimizer = build
    training.setup_model_and_optimizer = setup
    if os.environ.get('B1_GATE_STOP_AFTER_ROLLING') == '1':
        original_save = training.save_checkpoint_and_time
        def gate_save(*args, **kwargs):
            values = inspect.signature(original_save).bind(*args, **kwargs).arguments
            result = original_save(*args, **kwargs)
            if values.get('non_persistent_ckpt'):
                rank = torch.distributed.get_rank()
                out = Path(os.environ['TRANSITION_OUT']).parent/'saved-rolling-audit'
                out.mkdir(parents=True, exist_ok=True)
                (out/f'rank-{rank}.json').write_text(json.dumps(dict(iteration=values['iteration'],
                    moments=moment_evidence(values['optimizer'])), indent=2)+'\n')
                torch.distributed.barrier()
                if rank == 0: print('B1_GATE_ROLLING_SAVE_COMPLETE', flush=True)
                raise SystemExit(0)
            return result
        training.save_checkpoint_and_time = gate_save
    runpy.run_path(str(source/'pretrain_gpt.py'), run_name='__main__')


if __name__ == '__main__':
    main()
