"""Retained-state B1 audit plus explicit TWEO config/rolling-save receipts."""
import inspect
import functools
import json
import os
from pathlib import Path
import sys


def main():
    import torch
    import megatron.training.training as training
    from megatron.core.utils import get_model_config
    import conservative_b1_training as retained
    contract=json.loads(Path(os.environ['B1_RESTORE_CONTRACT']).read_text())
    fields=('tweo_loss_coeff','tweo_tau','tweo_start_step','tweo_warmup_steps','tweo_implementation','tweo_diagnostics_interval')
    if any('--'+key.replace('_','-') in sys.argv for key in fields):
        raise ValueError('Duplicate TWEO configuration')
    for key in fields:
        sys.argv.extend(['--'+key.replace('_','-'),str(contract['args'][key])])
    original_setup=training.setup_model_and_optimizer
    original_save=training.save_checkpoint_and_time
    out=Path(os.environ['TWEO_RECEIPT_OUT']);out.mkdir(parents=True,exist_ok=True)
    def setup(*args,**kwargs):
        result=original_setup(*args,**kwargs)
        for part in result[0]:
            cfg=get_model_config(part)
            retained.check_values({k:getattr(cfg,k) for k in fields},{k:contract['args'][k] for k in fields})
        return result
    @functools.wraps(original_save)
    def save(*args,**kwargs):
        values=inspect.signature(original_save).bind(*args,**kwargs).arguments
        result=original_save(*args,**kwargs)
        # The synchronous gate makes these exact saved-state receipts. Pilot
        # completion also requires native checkpoint metadata/shard validation.
        rank=torch.distributed.get_rank()
        receipt=dict(passed=True,rank=rank,iteration=values['iteration'],
                     non_persistent=bool(values.get('non_persistent_ckpt')),
                     moments=retained.moment_evidence(values['optimizer']),
                     tweo={k:contract['args'][k] for k in fields})
        path=out/f'training-rank{rank}.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(receipt)+'\n');tmp.replace(path)
        return result
    training.setup_model_and_optimizer=setup
    training.save_checkpoint_and_time=save
    retained.main()


if __name__=='__main__':main()
