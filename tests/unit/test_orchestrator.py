import json
from pathlib import Path

from oellm_autoexp.orchestrator import build_execution_plan, render_scripts, submit_jobs
from oellm_autoexp.slurm.client import FakeSlurmClient, FakeSlurmClientConfig
from oellm_autoexp.utils.start_condition import StartConditionResult


CONFIG_TEMPLATE = """
project:
  name: demo
  base_output_dir: {base_output}
sweep:
  axes:
    lr: [0.1, 0.01]
slurm:
  template_path: {template_path}
  script_dir: {script_dir}
  log_dir: {log_dir}
  client:
    class_name: FakeSlurmClient
monitoring:
  implementation:
    class_name: NullMonitor
backend:
  implementation:
    class_name: NullBackend
restart_policies:
  - mode: success
    implementation:
      class_name: NoRestartPolicy
"""

GATING_TEMPLATE = """
project:
  name: gate
  base_output_dir: {base_output}
sweep:
  axes:
    job.start_condition_cmd: ["echo 1"]
    job.start_condition_interval_seconds: [30]
    monitoring.termination_string: ["Done"]
slurm:
  template_path: {template_path}
  script_dir: {script_dir}
  log_dir: {log_dir}
  client:
    class_name: FakeSlurmClient
monitoring:
  implementation:
    class_name: NullMonitor
backend:
  implementation:
    class_name: NullBackend
restart_policies:
  - mode: success
    implementation:
      class_name: NoRestartPolicy
"""


def test_build_execution_plan_and_render(tmp_path: Path) -> None:
    base_output = tmp_path / "outputs"
    template_path = tmp_path / "template.sbatch"
    script_dir = tmp_path / "scripts"
    log_dir = tmp_path / "logs"

    template_path.write_text("#!/bin/bash\n{sbatch_directives}\n\nsrun {srun_opts}{launcher_cmd}\n")

    config_text = CONFIG_TEMPLATE.format(
        base_output=base_output,
        template_path=template_path,
        script_dir=script_dir,
        log_dir=log_dir,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text)

    plan = build_execution_plan(config_path)
    artifacts = render_scripts(plan)

    assert len(plan.jobs) == 2
    assert len(artifacts.job_scripts) == 2
    for script in artifacts.job_scripts:
        assert script.exists()
        content = script.read_text()
        assert "#SBATCH --job-name=" in content


def test_render_scripts_with_custom_slurm(tmp_path: Path) -> None:
    template_path = tmp_path / "template.sbatch"
    template_path.write_text("#!/bin/bash\n{sbatch_directives}\n\n{env_exports}\n\nsrun {srun_opts}{launcher_cmd}\n")

    script_dir = tmp_path / "scripts"
    log_dir = tmp_path / "logs"

    config_text = f"""
project:
  name: demo
  base_output_dir: {tmp_path / "outputs"}
sweep:
  axes:
    lr: [0.1]
slurm:
  template_path: {template_path}
  script_dir: {script_dir}
  log_dir: {log_dir}
  launcher_cmd: "module load foo &&"
  srun_opts: "--ntasks=4"
  env:
    ENV: 1
  sbatch_overrides:
    nodes: 2
  sbatch_extra_directives:
    - "#SBATCH --constraint=volta"
  client:
    class_name: FakeSlurmClient
monitoring:
  implementation:
    class_name: NullMonitor
backend:
  implementation:
    class_name: NullBackend
restart_policies:
  - mode: success
    implementation:
      class_name: NoRestartPolicy
"""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text)

    plan = build_execution_plan(config_path)
    artifacts = render_scripts(plan)

    script_path = artifacts.job_scripts[0]
    content = script_path.read_text()
    assert "#SBATCH --nodes=2" in content
    assert "#SBATCH --constraint=volta" in content
    assert "export ENV=1" in content
    assert "srun --ntasks=4 module load foo &&" in content


def test_submit_jobs_honours_start_condition(monkeypatch, tmp_path: Path) -> None:
    base_output = tmp_path / "outputs"
    template_path = tmp_path / "template.sbatch"
    script_dir = tmp_path / "scripts"
    log_dir = tmp_path / "logs"

    template_path.write_text("#!/bin/bash\n{sbatch_directives}\n")

    config_text = GATING_TEMPLATE.format(
        base_output=base_output,
        template_path=template_path,
        script_dir=script_dir,
        log_dir=log_dir,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text)

    plan = build_execution_plan(config_path)
    artifacts = render_scripts(plan)

    calls: list[tuple[str, int]] = []

    def fake_wait(command: str, *, interval_seconds=None, logger=None, sleep_fn=None):
        calls.append((command, interval_seconds))
        return StartConditionResult(True, "1", "", 0)

    monkeypatch.setattr("oellm_autoexp.orchestrator.wait_for_start_condition", fake_wait)

    controller = submit_jobs(plan, artifacts)

    assert calls == [("echo 1", 30)]

    job_state = next(iter(controller.jobs()))
    assert job_state.registration.start_condition_cmd == "echo 1"
    assert job_state.registration.termination_string == "Done"


def test_render_scripts_creates_array_assets(tmp_path: Path) -> None:
    base_output = tmp_path / "outputs"
    template_path = tmp_path / "template.sbatch"
    script_dir = tmp_path / "scripts"
    log_dir = tmp_path / "logs"

    template_path.write_text("#!/bin/bash\n{sbatch_directives}\n\n{env_exports}\n\n{launcher_cmd}\n")

    config_text = f"""
project:
  name: demo
  base_output_dir: {base_output}
sweep:
  axes:
    lr: [0.1, 0.01]
slurm:
  template_path: {template_path}
  script_dir: {script_dir}
  log_dir: {log_dir}
  array: true
  client:
    class_name: FakeSlurmClient
monitoring:
  class_name: NullMonitor
backend:
  class_name: NullBackend
restart_policies: []
"""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text)

    plan = build_execution_plan(config_path)
    artifacts = render_scripts(plan)

    assert artifacts.array_script is not None
    assert artifacts.array_script.exists()
    assert artifacts.sweep_json is not None
    payload = json.loads(artifacts.sweep_json.read_text())
    assert payload["project"] == "demo"
    assert len(payload["jobs"]) == 2
    assert payload["jobs"][0]["launch"]["argv"]


def test_submit_jobs_uses_array_submission(monkeypatch, tmp_path: Path) -> None:
    base_output = tmp_path / "outputs"
    template_path = tmp_path / "template.sbatch"
    script_dir = tmp_path / "scripts"
    log_dir = tmp_path / "logs"

    template_path.write_text("#!/bin/bash\n{sbatch_directives}\n\n{launcher_cmd}\n")

    config_text = f"""
project:
  name: arr
  base_output_dir: {base_output}
sweep:
  axes:
    lr: [0.1, 0.01]
slurm:
  template_path: {template_path}
  script_dir: {script_dir}
  log_dir: {log_dir}
  array: true
  client:
    class_name: FakeSlurmClient
monitoring:
  class_name: NullMonitor
backend:
  class_name: NullBackend
restart_policies: []
"""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text)

    plan = build_execution_plan(config_path)
    artifacts = render_scripts(plan)

    fake_client = FakeSlurmClient(FakeSlurmClientConfig())
    fake_client.configure(plan.config.slurm)

    call_args: list[tuple[str, Path]] = []

    def fake_submit_array(name, script_path, log_paths, task_names):
        call_args.append((name, script_path))
        return FakeSlurmClient.submit_array(fake_client, name, script_path, log_paths, task_names)

    monkeypatch.setattr(fake_client, "submit_array", fake_submit_array)

    controller = submit_jobs(plan, artifacts, fake_client)

    assert call_args
    assert call_args[0][1] == artifacts.array_script
    assert len(controller.jobs()) == 2


def test_submit_jobs_persists_and_restores(tmp_path: Path) -> None:
    base_output = tmp_path / "outputs"
    template_path = tmp_path / "template.sbatch"
    script_dir = tmp_path / "scripts"
    log_dir = tmp_path / "logs"

    template_path.write_text("#!/bin/bash\n{sbatch_directives}\n\n{launcher_cmd}\n")

    config_text = CONFIG_TEMPLATE.format(
        base_output=base_output,
        template_path=template_path,
        script_dir=script_dir,
        log_dir=log_dir,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text)

    plan = build_execution_plan(config_path)
    artifacts = render_scripts(plan)

    fake_client = FakeSlurmClient(FakeSlurmClientConfig())
    fake_client.configure(plan.config.slurm)

    submit_jobs(plan, artifacts, fake_client)

    state_file = plan.runtime.state_dir / "monitor" / "state.json"
    assert state_file.exists()
    data = json.loads(state_file.read_text())
    assert len(data.get("jobs", [])) == len(plan.jobs)

    # Simulate process restart by using a new client and re-submitting.
    fake_client_restarted = FakeSlurmClient(FakeSlurmClientConfig())
    fake_client_restarted.configure(plan.config.slurm)
    controller = submit_jobs(plan, artifacts, fake_client_restarted)

    assert len(controller.jobs()) == len(plan.jobs)
    assert len(fake_client_restarted.squeue()) == len(plan.jobs)
