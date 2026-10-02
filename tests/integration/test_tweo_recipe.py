"""Portable TWEO profile, native CLI rendering and post-load override rejection."""
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).parent))
from test_b1_recipes import test_resolved_b1_recipe as resolved_b1_recipe, isolated_config_cache
from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.config.schema import ConfigSetup

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import b1_run
import b1_tweo_training
import conservative_b1_training


def test_tweo_resolved_b1_and_save_settings(tmp_path,monkeypatch):
    resolved_b1_recipe('tweo',tmp_path,monkeypatch)
    recipe=json.loads((ROOT/'config/experiments/oellm_32b_dense/b1_reborn_tweo.yaml').read_text().split('\n',1)[1])
    assert recipe['backend']['launcher_script']=='${aux.repo_root}/tools/b1_tweo_training.py'
    assert recipe['backend']['megatron']==dict(async_save=False,use_persistent_ckpt_worker=False)


def test_profile_preserves_absolute_coefficient_ramp_on_resume():
    initial=b1_run.contract('tweo',80000,100000)
    resume=b1_run.contract('tweo',90750,100000)
    fields={k:v for k,v in initial['args'].items() if k.startswith('tweo_')}
    assert fields==dict(tweo_loss_coeff=7.817552855400486e-14,tweo_tau=3.,tweo_start_step=80000,
        tweo_warmup_steps=100,tweo_implementation='fused',tweo_diagnostics_interval=100)
    assert {k:resume['args'][k] for k in fields}==fields
    assert resume['floor']==1 and resume['args']['no_load_optim'] is False
    with pytest.raises(ValueError,match='predates'):b1_run.contract('tweo',60000,100000)


@pytest.mark.parametrize('wrong',['tweo_loss_coeff','tweo_implementation','tweo_diagnostics_interval',None])
def test_constructed_model_override_rejected(tmp_path,monkeypatch,wrong):
    contract=b1_run.contract('tweo',80000,100000)
    path=tmp_path/'contract.json';path.write_text(json.dumps(contract))
    monkeypatch.setenv('B1_RESTORE_CONTRACT',str(path));monkeypatch.setenv('TWEO_RECEIPT_OUT',str(tmp_path/'receipts'))
    monkeypatch.setattr(sys,'argv',['training.py'])
    values={k:v for k,v in contract['args'].items() if k.startswith('tweo_')}
    if wrong:values[wrong]='unexpected'
    fake=ModuleType('megatron.training.training');fake.setup_model_and_optimizer=lambda:([SimpleNamespace(**values)],None,None)
    fake.save_checkpoint_and_time=lambda **kw:None
    mega=ModuleType('megatron');training=ModuleType('megatron.training');core=ModuleType('megatron.core');utils=ModuleType('megatron.core.utils')
    mega.training=training;training.training=fake;mega.core=core;core.utils=utils;utils.get_model_config=lambda p:p
    for name,module in [('torch',ModuleType('torch')),('megatron',mega),('megatron.training',training),('megatron.training.training',fake),('megatron.core',core),('megatron.core.utils',utils)]:monkeypatch.setitem(sys.modules,name,module)
    monkeypatch.setattr(conservative_b1_training,'main',lambda:fake.setup_model_and_optimizer())
    if wrong:
        with pytest.raises(ValueError,match='B1_POST_LOAD_CONTRACT_FAILED'):b1_tweo_training.main()
    else:
        b1_tweo_training.main()
        for key,value in values.items():
            flag='--'+key.replace('_','-');assert sys.argv[sys.argv.index(flag)+1]==str(value)


def test_source_matches_validated_seven_file_overlay():
    manifest=json.loads((ROOT/'versions/tweo_source.json').read_text())
    source=ROOT/'submodules/Megatron-LM'
    assert (source/'pretrain_gpt.py').is_file(),'Initialize released Megatron before testing'
    for rel,digest in manifest['files'].items():
        assert hashlib.sha256((source/rel).read_bytes()).hexdigest()==digest,rel
