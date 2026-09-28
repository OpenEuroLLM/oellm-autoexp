"""The Complex-KDA fine-grained sweep, planned end to end (Hydra + DAG resolution).

Checks the wiring a dry run cannot show at a glance: every cooldown loads the
trunk of its own arm, size, ladder, gbs and lr at its branch step; every contcd
resumes its own firstcd directory; names are unique; widths match the matched
table. Offline, ~70s.
"""

from __future__ import annotations

import os
from collections import Counter

import pytest
from compoconf import asdict
from omegaconf import OmegaConf

from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.config.schema import ConfigSetup
from oellm_autoexp.orchestrator import build_execution_plan

CONFIG = "experiments/korbi/complex_kda/ckda_ladder_finegrained"


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    out = tmp_path_factory.mktemp("ckda_out")
    # One resolver worker: the fork pool hangs (and can segfault) inside a
    # pytest process, while it runs fine from scripts/run_autoexp.py.
    env = {
        "OUTPUT_DIR": str(out),
        "HF_HOME": "/dummy/hf_home",
        "HYDRA_STAGED_SWEEP_WORKERS": "1",
    }
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        setup = ConfigSetup(
            pwd=os.path.abspath("."), config_name=CONFIG, config_dir="config", overrides=[]
        )
        yield build_execution_plan(load_config_reference(config_setup=setup), config_setup=setup)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_plan_shape(plan):
    stages = Counter(j.config.stage.split("_")[0] for j in plan.jobs)
    # 6 arms x (5 chains; 2 + 2 + 1 + 3 + 4 = 12 budgets)
    assert stages == {"stable": 30, "firstcd": 72, "contcd": 72}
    names = Counter(j.config.job.name for j in plan.jobs)
    assert max(names.values()) == 1


def test_every_stage_links_to_its_own_chain(plan):
    for job in plan.jobs:
        c, m = job.config, job.config.backend.megatron
        stage = c.stage.split("_")[0]
        prefix = c.job.name.rsplit("_" + c.stage, 1)[0]
        base = os.path.dirname(c.job.base_output_dir)
        assert m.ffn_hidden_size == c.aux["ffn_by_size"][c.aux["size"]]
        if stage == "stable":
            assert m.load == c.job.base_output_dir and m.ckpt_step is None
            assert m.train_iters == max(m.save_extra_steps)
        elif stage == "firstcd":
            assert m.load.rstrip("/") == f"{base}/{prefix}_stable"
            assert m.ckpt_step == c.aux["start_iter"] == m.train_iters - 1
            assert m.save == c.job.base_output_dir
            cond = c.job.start_condition.command
            assert f"{base}/{prefix}_stable/latest_checkpointed_iteration.txt" in cond
            assert cond.endswith(f"-ge {c.aux['start_iter']}")
        else:
            firstcd = f"{base}/{prefix}_firstcd_decay{c.aux['budget_tag']}BT"
            assert m.load == m.save == firstcd and m.ckpt_step is None
            assert m.lr_wsd_decay_iters == m.train_iters - c.aux["start_iter"]
            assert c.job.start_condition.command.endswith(f"-ge {c.aux['start_iter'] + 1}")


def test_runtime_estimate_resolves(plan):
    total = 0.0
    for job in plan.jobs:
        hours = OmegaConf.select(OmegaConf.create(asdict(job.config)), "aux.true_time")
        assert float(hours) >= 0
        total += (
            float(hours) * job.config.slurm.sbatch.nodes * job.config.slurm.sbatch.gpus_per_node
        )
    assert 200 < total < 500  # BUDGET.md: 356 GPU-h
