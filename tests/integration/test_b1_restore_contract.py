"""Restore must retain Adam tensors/age while adopting the B1 hyperparameters."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
from conservative_b1_training import check_values, group_metadata, restore_group_settings
from gradient_transition_warmup import attach_warmup


def test_reapply_recipe_preserves_parameter_and_age():
    parameter = object()
    group = dict(params=[parameter], step=60000, lr=1.76e-4, weight_decay=.05,
                 betas=(.9, .95), eps=1e-8, wd_mult=1.)
    desired = dict(group_metadata(group), lr=3e-4, weight_decay=.1)
    changes = restore_group_settings([group], [((id(parameter),), desired)])
    assert group['params'][0] is parameter and group['step'] == 60000
    assert group['lr'] == 3e-4 and group['weight_decay'] == .1
    assert changes[0]['lr']['loaded'] == 1.76e-4
    with pytest.raises(ValueError):
        restore_group_settings([group], [((id(object()),), desired)])


def test_checkpoint_overrides_are_rejected():
    with pytest.raises(ValueError, match='POST_LOAD_CONTRACT_FAILED'):
        check_values(dict(lr=2e-4, output_z_loss_coeff=1e-4),
                     dict(lr=3e-4, output_z_loss_coeff=1e-5))
    check_values(dict(no_load_optim=None, no_load_rng=None),
                 dict(no_load_optim=False, no_load_rng=False))


class Scheduler:
    def __init__(self, steps, base):
        self.num_steps, self.base = steps, base
        self.group = {}
    def get_lr(self, group):
        return self.base
    def step(self, increment):
        self.num_steps += increment
        self.group['lr'] = self.get_lr(self.group)


def test_optional_ramp_keeps_original_anchor_on_resume_and_preserves_cooldown():
    anchor = 60000*4096
    scheduler = Scheduler(anchor+5000*4096, 3e-4)
    attach_warmup(scheduler, 4096, 10000, .1, anchor)
    assert scheduler.group['lr'] == pytest.approx(3e-4*.55)
    scheduler.step(5000*4096)
    assert scheduler.group['lr'] == pytest.approx(3e-4)
    cooldown = Scheduler(850000*4096, 1e-4)
    attach_warmup(cooldown, 4096, 1, 1., anchor)
    assert cooldown.group['lr'] == pytest.approx(1e-4)
