from pathlib import Path
import pytest

from monitor.loop import MonitorLoop, JobFileStore, JobRecordConfig, JobRuntimeConfig
from oellm_autoexp.monitor.adapter import SlurmClientAdapter
from oellm_autoexp.orchestrator import (
    build_execution_plan,
    render_scripts,
    _convert_to_registration,
)
from oellm_autoexp.slurm.client import FakeSlurmClient, FakeSlurmClientConfig
from oellm_autoexp.config.loader import load_config_reference

QUEUE_CONFIG = """
project:
  name: demo_${{index}}
  base_output_dir: {base_output}
  log_path: {base_output}/test.log
  log_path_current: {base_output}/test.log
sweep:
  type: product
  groups:
    - params:
        backend.dummy: [0, 1]
slurm:
  template_path: {template_path}
  script_dir: {script_dir}
  log_dir: {log_dir}
  array: false
  launcher_cmd: ""
  srun_opts: ""
  client:
    class_name: FakeSlurmClient
monitoring:
  class_name: SlurmLogMonitor
  check_interval_seconds: 5
  log_path: ${{project.log_path_current}}
  log_events:
    - name: checkpoint_ready
      pattern: 'CHECKPOINT (?P<ckpt>\\S+)'
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
  base_command: ["echo", "1"]
index: 0
"""


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


def test_run_autoexp_action_execution(tmp_path: Path, monkeypatch) -> None:
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

    session_dir = tmp_path / "monitor_state" / "queue-test"
    store = JobFileStore(session_dir)
    slurm = FakeSlurmClient(FakeSlurmClientConfig())
    adapter = SlurmClientAdapter(slurm)
    loop = MonitorLoop(store, adapter, poll_interval_seconds=60)

    job_plan = plan.jobs[0]

    # Register job in store
    reg = _convert_to_registration(job_plan, plan.runtime.monitor.config, artifacts.job_scripts[0])
    job_record = JobRecordConfig(
        job_id=job_plan.name,  # Use name as ID for test
        registration=reg,
        runtime=JobRuntimeConfig(submitted=False),
    )
    store.upsert(job_record)

    # Start the job (simulate submission)
    # MonitorLoop would do this if we call observe_once, but we can fast track
    loop._start_job(job_record)
    store.upsert(job_record)
    slurm_job_id = job_record.runtime.runtime_job_id

    # Ensure job is RUNNING in fake slurm
    slurm.set_state(slurm_job_id, "RUNNING")

    # Mock _run_command to capture execution
    class DummyProc:
        def __init__(self, returncode: int):
            self.returncode = returncode
            self.stdout = ""
            self.stderr = "" if returncode == 0 else "error"

    responses = [DummyProc(1), DummyProc(0)]
    executed_commands = []

    def fake_run(cmd, cwd=None, env=None):
        executed_commands.append(cmd)
        if responses:
            return responses.pop(0)
        return DummyProc(0)

    monkeypatch.setattr("oellm_autoexp.monitor.actions._run_command", fake_run)

    # Write log content to trigger checkpoint
    # log_path = Path(
    #     job_plan.log_path
    # )  # In fake slurm with name as ID, log path logic might be tricky?
    # Actually _convert_to_registration sets log_path.
    # But MonitorLoop resolves log path using _resolve_log_path.

    # For FakeSlurm, submit uses the log path we passed.
    # In _convert_to_registration, we passed job_plan.log_path.
    # So we write to that path.
    # But wait, log_path in config has %j.
    # FakeSlurm doesn't expand %j in file system, but it uses the path provided.
    # orchestrator.submit_jobs does _create_current_log_symlink.

    # Here we are manually setting up.
    # job_plan.log_path contains %j.
    # We should use expanded path.

    # Let's check what loop._resolve_log_path does.
    # It uses resolve_log_path(job.registration.log_path, job.runtime.runtime_job_id)

    # Ensure log directory exists
    Path(job_plan.log_path).parent.mkdir(parents=True, exist_ok=True)

    # Create the log file that loop will look for
    # We need to know what path loop resolves to.
    expanded_log_path = Path(job_plan.log_path.replace("%j", slurm_job_id))
    expanded_log_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_one = Path(job_plan.output_dir) / "chkpts" / "iter_1.pt"
    expanded_log_path.write_text(f"iter 1\nCHECKPOINT {ckpt_one}\n")

    # Run loop observation
    loop.observe_once()

    # Verify action execution
    assert len(executed_commands) == 1
    cmd = executed_commands[0]
    # Check if command contains expected args
    # cmd is list of strings
    assert "scripts/run_autoexp.py" in cmd[1]  # [0] is python executable
    assert any(str(ckpt_one) in arg for arg in cmd)

    # Append second checkpoint
    ckpt_two = Path(job_plan.output_dir) / "chkpts" / "iter_2.pt"
    with open(expanded_log_path, "a") as f:
        f.write(f"CHECKPOINT {ckpt_two}\n")

    loop.observe_once()

    assert len(executed_commands) == 2
    cmd2 = executed_commands[1]
    assert any(str(ckpt_two) in arg for arg in cmd2)
