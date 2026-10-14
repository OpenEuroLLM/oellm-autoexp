"""Resolve both collaborator recipes through the real loader and CLI renderer."""
import json
from pathlib import Path
import sys

import pytest
from oellm_autoexp.backends.megatron_backend import MegatronBackend
from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.config.schema import ConfigSetup

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import b1_run


@pytest.fixture(autouse=True)
def isolated_config_cache():
    from oellm_autoexp.hydra_staged_sweep.config.cache import clear
    clear()
    yield
    clear()


@pytest.mark.parametrize('variant',['legacy','patched'])
def test_resolved_b1_recipe(variant,tmp_path,monkeypatch):
    from oellm_autoexp.hydra_staged_sweep.config.cache import clear
    clear()
    monkeypatch.setenv('PROJECT_DIR',str(ROOT))
    monkeypatch.setenv('JUPITER_EXCLUDE_NODES',str(tmp_path/'exclude'))
    values=dict(run_root=str(tmp_path/'run'),repo_root='/portable/autoexp',source=str(tmp_path/'Megatron-LM'),
        data_manifest=str(tmp_path/'data'),data_cache=str(tmp_path/'cache'),tokenizer=str(tmp_path/'tokenizer'),
        remaining_steps='40000',image=str(tmp_path/'training.sif'),account='test-account',walltime='12:00:00',exclude_file=str(tmp_path/'exclude'))
    cfg=load_config_reference(config_setup=ConfigSetup(config_dir=ROOT/'config',
        config_name='experiments/oellm_32b_dense/b1_reborn_'+variant,
        overrides=[f'aux.{k}={v}' for k,v in values.items()]+['backend.megatron.data_args_path=null']))
    m=cfg.backend.megatron
    spec=json.loads((ROOT/'versions/b1_recipes.json').read_text())
    for key,value in spec['args'].items(): assert getattr(m,key)==value,key
    assert cfg.aux['expected_source']==spec['variants'][variant]['commit']
    assert m.save_interval==2000 and m.non_persistent_save_interval==125
    assert m.exit_duration_in_mins is None
    assert m.exit_on_missing_checkpoint and m.log_params_norm and m.log_num_zeros_in_grad
    assert cfg.slurm.sbatch.signal == 'B:USR1@240'
    assert m.lr_warmup_samples==40960000
    assert cfg.slurm.sbatch.time=='12:00:00'
    assert cfg.container.python=='/opt/venv/bin/python'
    assert cfg.job.checkpoint_hook_command==''
    command=MegatronBackend(cfg.backend).build_launch_command()
    assert '--defer-embedding-wgrad-compute' in command
    assert '--overlap-grad-reduce' in command
    assert '--decoupled-lr' not in command
    assert '--lr 0.0003' in command and '--weight-decay 0.1' in command
    assert '--output-z-loss-coeff 1e-05' in command
    assert '--exit-duration-in-mins' not in command
    for value in values.values(): command=command.replace(value,'<provided>')
    assert 'jitsev1' not in command and 'revival_1' not in command


def test_explicit_retained_state_and_optional_ramp():
    for variant in ('legacy','patched'):
        c=b1_run.contract(variant,60000,100000)
        assert c['args']['no_load_optim'] is False and c['args']['no_load_rng'] is False
        assert c['args']['use_checkpoint_args'] is False
        assert c['floor']==1 and c['warmup_updates']==1
        ramp=b1_run.contract(variant,60000,100000,10000,.1)
        assert ramp['anchor_samples']==60000*4096
        assert ramp['warmup_updates']==10000 and ramp['floor']==.1
    with pytest.raises(ValueError): b1_run.contract('patched',60000,60000)


@pytest.mark.parametrize('path',['/tmp/a;bad','/tmp/a b','/tmp/$(bad)','/tmp/a,b'])
def test_unsafe_paths_rejected(path):
    with pytest.raises(ValueError): b1_run.safe_path(path)
