"""Checkpoint hook (2026-09-06): the `checkpoint_saved` log event runs the job's checkpoint_hook_command when a
persistent checkpoint's iteration is a multiple of checkpoint_hook_every (IterationMultipleCondition)."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from oellm_autoexp.monitor.actions import LogEvent, LogEventConfig, RunCommandAction, RunCommandActionConfig, ActionContext, EventRecord
from oellm_autoexp.monitor.conditions import ConditionContext, IterationMultipleCondition, IterationMultipleConditionConfig

POLICIES = [Path(__file__).resolve().parents[2] / "config" / "job" / f for f in ("auto_restart_ckptreset_faultscan.yaml", "auto_restart_ckptreset.yaml")]
PERSISTENT = "[default0]:  [2026-09-06 11:19:24.346793] successfully saved checkpoint from iteration   80000 to /e/fscratch/e-sta-openeurollm/production/32b_dense_revival_1_body-3e-4_head-2e-4/training_ckpts/checkpoints in torch_dist format"
ROLLING = "[default0]:  [2026-09-06 15:23:00.1] successfully saved checkpoint from iteration   83375 to /e/fscratch/e-sta-openeurollm/production/32b_dense_revival_1_body-3e-4_head-2e-4/training_ckpts/checkpoints_rolling in torch_dist format"


def event_cfg(policy):
    cfg = yaml.safe_load(policy.read_text())
    return next(e for e in cfg["log_events"] if e["name"] == "checkpoint_saved")


def test_event_present_in_both_policies_with_a_run_command():
    for p in POLICIES:
        e = event_cfg(p)
        assert e["action"]["class_name"] == "RunCommandAction"
        assert e["condition"]["class_name"] == "IterationMultipleCondition"
        assert e["extract_groups"] == {"iteration": "iteration", "ckpt_dir": "ckpt_dir"}


def test_pattern_matches_persistent_saves_only():
    pat = event_cfg(POLICIES[0])["pattern"]
    m = re.search(pat, PERSISTENT)
    assert m and m.group("iteration") == "80000" and m.group("ckpt_dir").endswith("/training_ckpts/checkpoints")
    assert re.search(pat, ROLLING) is None


def test_log_event_extracts_the_groups():
    e = event_cfg(POLICIES[0])
    cfg = LogEventConfig(name="checkpoint_saved", pattern=e["pattern"], pattern_type="regex", extract_groups=e["extract_groups"], action=RunCommandActionConfig(command="x"))
    trig = LogEvent(cfg).check_triggers(PERSISTENT + "\n" + ROLLING)
    assert len(trig) == 1 and trig[0]["iteration"] == "80000" and trig[0]["ckpt_dir"].endswith("/checkpoints")


def test_iteration_multiple_condition():
    c = IterationMultipleCondition(IterationMultipleConditionConfig(every=4000))
    assert c.check(ConditionContext(job_metadata={"iteration": "80000"})).passed
    assert not c.check(ConditionContext(job_metadata={"iteration": "86000"})).passed
    assert not c.check(ConditionContext(job_metadata={})).passed
    assert not IterationMultipleCondition(IterationMultipleConditionConfig(every=0)).check(ConditionContext(job_metadata={"iteration": "80000"})).passed


def test_run_command_is_templated_with_iteration_and_ckpt_dir(tmp_path):
    out = tmp_path / "hook.txt"
    act = RunCommandAction(RunCommandActionConfig(command=f"echo {{iteration}} {{ckpt_dir}} > {out}", timeout_s=30))
    ev = EventRecord(event_id="e", name="checkpoint_saved", source="log", payload={"iteration": "84000", "ckpt_dir": "/x/checkpoints"}, metadata={"job_id": "j"})
    res = act.execute(ActionContext(event=ev, job_metadata={}))
    assert res.status == "success" and out.read_text().strip() == "84000 /x/checkpoints"
