from pathlib import Path

import pytest
from compoconf import parse_config

from oellm_autoexp.monitor.action_queue import ActionQueue
from oellm_autoexp.monitor.actions import ActionContext, BaseMonitorAction
from oellm_autoexp.monitor.controller import JobRegistration, MonitorController
from oellm_autoexp.monitor.watcher import (
    SlurmLogMonitor,
    SlurmLogMonitorConfig,
)
from oellm_autoexp.orchestrator import build_execution_plan, render_scripts
from oellm_autoexp.persistence import MonitorStateStore
from oellm_autoexp.slurm.client import FakeSlurmClient, FakeSlurmClientConfig
from oellm_autoexp.config.loader import load_config_reference

CONFIG = """
project:
  name: demo
  base_output_dir: {base_output}
sweep:
  axes:
    lr: [0.1]
slurm:
  template_path: {template_path}
  script_dir: {script_dir}
  log_dir: {log_dir}
  launcher_cmd: ""
  srun_opts: ""
  client:
    class_name: FakeSlurmClient
monitoring:
  class_name: SlurmLogMonitor
  poll_interval_seconds: 10
  check_interval_seconds: 10
  state_events:
    - name: stall
      state:
        class_name: CrashState
      actions:
        - class_name: EventAction
          action:
            class_name: RestartAction
            reason: "stall restart"
    - name: success
      state:
        class_name: SuccessState
backend:
  class_name: NullBackend
"""

QUEUE_CONFIG = """
project:
  name: demo
  base_output_dir: {base_output}
sweep:
  axes:
    only: [0]
slurm:
  template_path: {template_path}
  script_dir: {script_dir}
  log_dir: {log_dir}
  launcher_cmd: ""
  srun_opts: ""
  client:
    class_name: FakeSlurmClient
monitoring:
  class_name: SlurmLogMonitor
  poll_interval_seconds: 5
  check_interval_seconds: 5
  log_path_template: {log_dir}/queue-%j.log
  log_events:
    - name: checkpoint_ready
      pattern: "CHECKPOINT (?P<ckpt>\\\\S+)"
      pattern_type: regex
      metadata:
        kind: checkpoint
      extract_groups:
        checkpoint_path: ckpt
      actions:
        - class_name: EventAction
          mode: queue
          action:
            class_name: RunAutoexpAction
            script: scripts/run_autoexp.py
            config_path: "{{output_dir}}/provenance/config_reference.json"
            overrides:
              - "evaluation.checkpoint={{checkpoint_path}}"
  state_events:
    - name: success
      state:
        class_name: SuccessState
backend:
  class_name: NullBackend
"""


def test_fake_slurm_monitoring_cycle(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SLURM_ACCOUNT", "debug")
    base_output = tmp_path / "outputs"
    template_path = tmp_path / "template.sbatch"
    script_dir = tmp_path / "scripts"
    log_dir = tmp_path / "logs"
    template_path.write_text("#!/bin/bash\n{sbatch_directives}\n\nsrun {srun_opts}{launcher_cmd}\n")

    config_text = CONFIG.format(
        base_output=base_output,
        template_path=template_path,
        script_dir=script_dir,
        log_dir=log_dir,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text)

    plan = build_execution_plan(config_path)
    artifacts = render_scripts(plan)

    slurm = FakeSlurmClient(FakeSlurmClientConfig())
    controller = MonitorController(plan.runtime.monitor, slurm)

    job = plan.jobs[0]
    job_id = slurm.submit(job.name, artifacts.job_scripts[0], job.log_path)
    controller.register_job(
        job_id,
        JobRegistration(name=job.name, script_path=artifacts.job_scripts[0], log_path=job.log_path),
    )

    decision = controller.handle_state_change(job_id, "stall")
    assert decision.action == "restart"

    new_job_id = next(iter(controller.jobs())).job_id
    decision = controller.handle_state_change(new_job_id, "success")
    assert decision.action == "success"


def test_hydra_plan(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SLURM_ACCOUNT", "debug")
    monkeypatch.setenv("CONTAINER_CACHE_DIR", "debug")
    project_cfg = Path("config/project/default.yaml")
    slurm_cfg = Path("config/slurm/juwels.yaml")
    if not (project_cfg.exists() and slurm_cfg.exists()):
        pytest.skip("juwels config not available in this checkout")

    cfg = load_config_reference(
        "autoexp", Path("config"), overrides=["project=default", "slurm=juwels"]
    )
    plan = build_execution_plan(cfg)
    artifacts = render_scripts(plan)
    assert artifacts.job_scripts


def test_restart_on_cancelled_state(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SLURM_ACCOUNT", "debug")
    monitor_cfg = parse_config(
        SlurmLogMonitorConfig,
        {
            "poll_interval_seconds": 5,
            "check_interval_seconds": 5,
            "log_path_template": str(tmp_path / "logs" / "demo-%j.log"),
            "state_events": [
                {
                    "name": "crash",
                    "state": {"class_name": "CrashState"},
                    "actions": [
                        {
                            "class_name": "EventAction",
                            "action": {"class_name": "RestartAction", "reason": "crash restart"},
                        }
                    ],
                }
            ],
        },
    )
    monitor = SlurmLogMonitor(monitor_cfg)
    slurm = FakeSlurmClient(FakeSlurmClientConfig())
    controller = MonitorController(monitor, slurm)

    log_path = tmp_path / "logs" / "demo-%j.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")

    job_id = slurm.submit("demo", tmp_path / "demo.sbatch", str(log_path))
    registration = JobRegistration(
        name="demo", script_path=str(tmp_path / "demo.sbatch"), log_path=str(log_path)
    )
    controller.register_job(job_id, registration)

    slurm.set_state(job_id, "RUNNING")
    controller.observe_once_sync()

    slurm.set_state(job_id, "CANCELLED")
    result = controller.observe_once_sync()

    assert job_id not in [state.job_id for state in controller.jobs()]
    assert any(decision.action == "restart" for decision in result.decisions.values())


def _run_queue_worker(
    queue: ActionQueue,
    state_store: MonitorStateStore,
    expected_status: str,
) -> None:
    record = queue.claim_next()
    assert record is not None
    events = state_store.load_events()
    assert record.event_id in events
    event = events[record.event_id]
    cfg_dict = dict(record.config)
    cfg_dict.setdefault("class_name", record.action_class)
    action_cfg = parse_config(BaseMonitorAction.cfgtype, cfg_dict)
    action = action_cfg.instantiate(BaseMonitorAction)
    context = ActionContext(
        event=event,
        job_metadata=record.metadata.get("job", {}),
        workspace=None,
    )
    result = action.execute(context)
    assert result.status == expected_status
    action.update_event(event, result)
    state_store.upsert_event(event)
    queue.mark_done(
        record.queue_id,
        status="done" if result.status == "success" else "failed",
        result={"message": result.message},
    )


def test_run_autoexp_action_queue(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SLURM_ACCOUNT", "debug")
    base_output = tmp_path / "outputs"
    template_path = tmp_path / "template.sbatch"
    script_dir = tmp_path / "scripts"
    log_dir = tmp_path / "logs"
    template_path.write_text("#!/bin/bash\n{sbatch_directives}\n\nsrun {srun_opts}{launcher_cmd}\n")

    config_text = QUEUE_CONFIG.format(
        base_output=base_output,
        template_path=template_path,
        script_dir=script_dir,
        log_dir=log_dir,
    )
    config_path = tmp_path / "queue.yaml"
    config_path.write_text(config_text)

    plan = build_execution_plan(config_path)
    artifacts = render_scripts(plan)
    state_store = MonitorStateStore(tmp_path / "monitor_state", session_id="queue-test")
    slurm = FakeSlurmClient(FakeSlurmClientConfig())
    controller = MonitorController(plan.runtime.monitor, slurm, state_store)

    job = plan.jobs[0]
    job_id = slurm.submit(job.name, artifacts.job_scripts[0], job.log_path)
    registration = JobRegistration(
        name=job.name,
        script_path=artifacts.job_scripts[0],
        log_path=job.log_path,
        metadata={"parameters": dict(job.parameters), "output_dir": job.output_dir},
    )
    controller.register_job(job_id, registration)
    slurm.set_state(job_id, "RUNNING")

    queue_dir = state_store.session_path.with_suffix(".actions")
    queue = ActionQueue(queue_dir)

    log_path = controller._expand_log_path(job_id, job.log_path)  # type: ignore[attr-defined]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_one = Path(job.output_dir) / "chkpts" / "iter_1.pt"
    log_path.write_text(f"iter 1\nCHECKPOINT {ckpt_one}\n")

    controller.observe_once_sync()

    records = queue.list()
    assert len(records) == 1
    record = records[0]
    assert record.action_class == "RunAutoexpAction"
    assert record.config["config_path"].endswith("provenance/config_reference.json")
    assert str(ckpt_one) in record.config["overrides"][0]
    assert record.metadata["job"]["job_name"] == job.name

    class DummyProc:
        def __init__(self, returncode: int):
            self.returncode = returncode
            self.stdout = ""
            self.stderr = "" if returncode == 0 else "error"

    responses = [DummyProc(1), DummyProc(0)]

    def fake_run(cmd, cwd=None, env=None):
        return responses.pop(0)

    monkeypatch.setattr("oellm_autoexp.monitor.actions._run_command", fake_run)

    _run_queue_worker(queue, state_store, expected_status="failed")
    assert not queue.list()

    ckpt_two = Path(job.output_dir) / "chkpts" / "iter_2.pt"
    log_path.write_text(log_path.read_text() + f"CHECKPOINT {ckpt_two}\n")
    controller.observe_once_sync()

    second_record = queue.list()[0]
    assert str(ckpt_two) in second_record.config["overrides"][0]
    _run_queue_worker(queue, state_store, expected_status="success")
    assert not queue.list()
