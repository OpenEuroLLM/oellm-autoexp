"""Restart hooks: a RestartAction's ``pre_command`` runs before the resubmission
and ``exclude_file`` refreshes ``slurm.sbatch.exclude`` from the live exclusion
list (the stored SlurmConfig is otherwise frozen at plan time), see
``config/job/auto_restart_ckptreset_faultscan.yaml``."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir

from oellm_autoexp.monitor.actions import (
    ActionContext,
    EventRecord,
    RestartAction,
    RestartActionConfig,
    RunCommandAction,
    RunCommandActionConfig,
)
from oellm_autoexp.monitor.loop import MonitorLoop


def test_restart_action_carries_its_hooks_in_the_result():
    cfg = RestartActionConfig(reason="r", pre_command="echo {job_id}", exclude_file="/tmp/x")
    event = EventRecord(event_id="e", name="n", source="log")
    result = RestartAction(cfg).execute(ActionContext(event=event, job_metadata={"job_id": "j"}))
    assert result.special == "restart"
    assert result.metadata["pre_command"] == "echo {job_id}"
    assert result.metadata["exclude_file"] == "/tmp/x"


def test_run_command_action_renders_and_runs(tmp_path):
    out = tmp_path / "out.txt"
    cfg = RunCommandActionConfig(command=f"echo node={{node}} job={{job_id}} > {out}")
    event = EventRecord(event_id="e", name="n", source="log", payload={"node": "jpbo-001-01"})
    result = RunCommandAction(cfg).execute(ActionContext(event=event, job_metadata={"job_id": "j1"}))
    assert result.status == "success"
    assert out.read_text().strip() == "node=jpbo-001-01 job=j1"


def test_run_restart_hooks_runs_pre_command_and_refreshes_excludes(tmp_path):
    exclude = tmp_path / "exclude.txt"
    exclude.write_text("# reason\njpbo-001-01\n\njpbo-002-02  \n")
    marker = tmp_path / "scan.txt"
    loop = MonitorLoop.__new__(MonitorLoop)  # no store/clients needed for the hook itself
    loop._pending_restart_hooks = {
        "job-a": {
            "pre_command": f"echo scanned {{runtime_job_id}} {{log_path}} > {marker}",
            "pre_command_timeout_s": 30,
            "exclude_file": str(exclude),
        }
    }
    sbatch = SimpleNamespace(exclude="jpbo-001-01")
    definition = SimpleNamespace(metadata={}, name="jobname", class_name="SlurmJobConfig", slurm=SimpleNamespace(sbatch=sbatch), log_path=str(tmp_path / "slurm-%j.log"))
    job = SimpleNamespace(job_id="job-a", definition=definition, runtime=SimpleNamespace(runtime_job_id="4242", start_ts=1.0, attempts=1))
    # isinstance(job.definition, SlurmJobConfig) guards the exclude refresh: patch it for the namespace
    import oellm_autoexp.monitor.loop as loop_mod

    real = loop_mod.SlurmJobConfig
    loop_mod.SlurmJobConfig = SimpleNamespace  # type: ignore[assignment]
    try:
        loop._run_restart_hooks(job, "4242")
    finally:
        loop_mod.SlurmJobConfig = real
    assert marker.read_text().startswith("scanned 4242 ")
    assert "slurm-4242.log" in marker.read_text()
    assert sbatch.exclude == "jpbo-001-01,jpbo-002-02"
    assert "job-a" not in loop._pending_restart_hooks


def test_hooks_are_a_no_op_without_configuration():
    loop = MonitorLoop.__new__(MonitorLoop)
    loop._pending_restart_hooks = {}
    job = SimpleNamespace(job_id="j", definition=SimpleNamespace(metadata={}, name="n", class_name="c", slurm=SimpleNamespace(sbatch=SimpleNamespace(exclude="a"))), runtime=SimpleNamespace(runtime_job_id=None, start_ts=None, attempts=0))
    loop._run_restart_hooks(job, None)  # must not raise
    assert job.definition.slurm.sbatch.exclude == "a"


def test_faultscan_policy_hooks_every_restart_action():
    config_dir = Path(__file__).resolve().parents[2] / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="autoexp", overrides=["job=auto_restart_ckptreset_faultscan", "++job.restart_pre_command=scan-the-log", "++job.restart_exclude_file=/tmp/list"])
    restarts = [e for e in cfg.job.log_events if e["action"]["class_name"] == "RestartAction"]
    assert len(restarts) >= 10
    for e in restarts:
        assert e["action"]["pre_command"] == "scan-the-log"
        assert e["action"]["exclude_file"] == "/tmp/list"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        base = compose(config_name="autoexp", overrides=["job=auto_restart_ckptreset"])
    assert [e["name"] for e in cfg.job.log_events] == [e["name"] for e in base.job.log_events]


def test_restart_hook_variables_include_log_dir(tmp_path):
    """The pre_command may scan the run's whole log directory ({log_dir}), not only the failed job's log:
    the lines that name a faulty node are written after the kill (2026-09-06, jobs 1692960/1693057)."""
    import oellm_autoexp.monitor.loop as loop_mod
    from oellm_autoexp.monitor.loop import JobRecord, JobRuntime, MonitorLoop
    loop = MonitorLoop.__new__(MonitorLoop)
    log = tmp_path / "logs" / "slurm-7.log"; log.parent.mkdir(); log.write_text("x")
    job = JobRecord(job_id="j", definition=SimpleNamespace(name="n", slurm=SimpleNamespace(sbatch=SimpleNamespace(exclude=""))), runtime=JobRuntime(submitted=True, runtime_job_id="7"))
    loop._pending_restart_hooks = {"j": {"pre_command": "echo {runtime_job_id} {log_dir}", "exclude_file": ""}}
    loop._build_job_metadata = lambda j: {}
    loop._resolve_log_path = lambda j: log
    import subprocess
    seen = {}; orig = subprocess.run
    subprocess.run = lambda cmd, **kw: (seen.setdefault("cmd", cmd), SimpleNamespace(returncode=0, stdout="", stderr=""))[1]
    try:
        loop._run_restart_hooks(job, "7")
    finally:
        subprocess.run = orig
    assert f"7 {log.parent}" in str(seen["cmd"])
