import json
from pathlib import Path

from oellm_autoexp.orchestrator import (
    build_execution_plan,
    load_monitor_controller,
    render_scripts,
    submit_jobs,
)
from oellm_autoexp.slurm.client import FakeSlurmClient, FakeSlurmClientConfig
from oellm_autoexp.utils.start_condition import StartConditionResult


CONFIG_TEMPLATE = """
project:
  name: demo_${{index}}
  base_output_dir: {base_output}
  log_path: {log_dir}/slurm-%j.out
  log_path_current: {log_dir}/current.log
sweep:
  grids:
    - backend.base_command: [["echo", "0"], ["echo", "1"]]
slurm:
  template_path: {template_path}
  script_dir: {script_dir}
  log_dir: {log_dir}
  client:
    class_name: FakeSlurmClient
monitoring:
  class_name: NullMonitor
  log_path: {log_dir}/current.log
backend:
  class_name: NullBackend
  base_command: ["echo", "Hello"]
index: 0  # will be replaced
"""

GATING_TEMPLATE = """
project:
  name: gate_${{index}}
  base_output_dir: {base_output}
  log_path: {log_dir}/slurm-%j.out
  log_path_current: {log_dir}/current.log
sweep:
  grids:
    - job.start_condition_cmd: ["echo 1"]
      job.start_condition_interval_seconds: [30]
      monitoring.termination_string: ["Done"]
slurm:
  template_path: {template_path}
  script_dir: {script_dir}
  log_dir: {log_dir}
  client:
    class_name: FakeSlurmClient
monitoring:
  class_name: NullMonitor
  log_path: {log_dir}/current.log
  termination_string: ""
job:
  start_condition_cmd: null
backend:
  class_name: NullBackend

index: 0 # will be replaced
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
        assert Path(script).exists()
        content = Path(script).read_text()
        assert "#SBATCH --job-name=" in content


def test_build_execution_plan_subset(tmp_path: Path) -> None:
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
    config_path = tmp_path / "config_subset.yaml"
    config_path.write_text(config_text)

    plan = build_execution_plan(config_path, subset_indices={1})
    assert len(plan.jobs) == 1
    assert plan.jobs[0].name.endswith("_1")


def test_render_scripts_with_custom_slurm(tmp_path: Path) -> None:
    template_path = tmp_path / "template.sbatch"
    template_path.write_text(
        "#!/bin/bash\n{sbatch_directives}\n\n{env_exports}\n\nsrun {srun_opts}{launcher_cmd}\n"
    )

    script_dir = tmp_path / "scripts"
    log_dir = tmp_path / "logs"

    config_text = f"""
project:
  name: demo_${{index}}
  base_output_dir: {tmp_path / "outputs"}
  log_path: {log_dir}/slurm-%j.out
  log_path_current: {log_dir}/current.log
sweep:
  grids:
    - backend.base_command: [["echo", "0"], ["echo", "1"]]
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
  class_name: NullMonitor
  log_path: {log_dir}/current.log
backend:
  class_name: NullBackend
  base_command: ["echo", "Hello"]
index: 0
"""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text)

    plan = build_execution_plan(config_path)
    artifacts = render_scripts(plan)

    script_path = artifacts.job_scripts[0]
    content = Path(script_path).read_text()
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

    submission = submit_jobs(plan, artifacts)
    controller = submission.controller
    assert submission.submitted_job_ids

    assert calls == [("echo 1", 30)]

    job_state = next(iter(controller.jobs()))
    assert job_state.registration.start_condition_cmd == "echo 1"
    assert job_state.registration.termination_string == "Done"


def test_render_scripts_creates_array_assets(tmp_path: Path) -> None:
    base_output = tmp_path / "outputs"
    template_path = tmp_path / "template.sbatch"
    script_dir = tmp_path / "scripts"
    log_dir = tmp_path / "logs"

    template_path.write_text(
        "#!/bin/bash\n{sbatch_directives}\n\n{env_exports}\n\n{launcher_cmd}\n"
    )

    config_text = f"""
project:
  name: demo_${{index}}
  base_output_dir: {base_output}
  log_path: {log_dir}/slurm-%j.out
  log_path_current: {log_dir}/current.log
sweep:
  grids:
    - backend.base_command: [["echo", "0"], ["echo", "1"]]
slurm:
  template_path: {template_path}
  script_dir: {script_dir}
  log_dir: {log_dir}
  array: true
  client:
    class_name: FakeSlurmClient
monitoring:
  class_name: NullMonitor
  log_path: {log_dir}/current.log
backend:
  class_name: NullBackend
  base_command: ["echo", "Hello"]
index: 0
"""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text)

    plan = build_execution_plan(config_path)
    artifacts = render_scripts(plan)

    assert artifacts.array_script is not None
    assert Path(artifacts.array_script).exists()
    assert artifacts.sweep_json is not None
    payload = json.loads(Path(artifacts.sweep_json).read_text())
    assert payload["project_name"] == "demo_0"
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
  name: arr_${{index}}
  base_output_dir: {base_output}
  log_path: {log_dir}/slurm-%j.out
  log_path_current: {log_dir}/current.log
sweep:
  grids:
    - backend.base_command: [["echo", "0"], ["echo", "1"]]
slurm:
  template_path: {template_path}
  script_dir: {script_dir}
  log_dir: {log_dir}
  array: true
  client:
    class_name: FakeSlurmClient
monitoring:
  class_name: NullMonitor
  log_path: {log_dir}/current.log
backend:
  class_name: NullBackend
  base_command: ["echo", "Hello"]

index: 0
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

    submission = submit_jobs(plan, artifacts, fake_client)
    controller = submission.controller
    assert set(submission.submitted_job_ids) == {state.job_id for state in controller.jobs()}

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

    submission = submit_jobs(plan, artifacts, fake_client)
    controller = submission.controller
    assert len(submission.submitted_job_ids) == len(plan.jobs)

    state_file = plan.runtime.state_dir / "monitor" / "state.json"
    assert state_file.exists()
    data = json.loads(state_file.read_text())
    assert len(data.get("jobs", [])) == len(plan.jobs)

    # Simulate process restart by loading the same monitoring session.
    fake_client_restarted = FakeSlurmClient(FakeSlurmClientConfig())
    fake_client_restarted.configure(plan.config.slurm)
    restored = load_monitor_controller(
        plan,
        fake_client_restarted,
        session_id=submission.session_id,
    )
    controller = restored.controller
    assert restored.submitted_job_ids == []

    assert len(controller.jobs()) == len(plan.jobs)
    assert len(fake_client_restarted.squeue()) == len(plan.jobs)


def test_hydra_group_overrides_in_sweep(tmp_path: Path) -> None:
    """Test that sweep parameters can use Hydra group overrides.

    This test demonstrates that job.parameters like "backend=variant1" are applied
    as Hydra overrides, allowing sweeps over:
    - Hydra group selections (e.g., backend: [variant1, variant2])
    - Nested config values (e.g., backend.megatron.lr: [0.1, 0.01])
    - Any other configuration parameter

    Note on escaping:
    - For values with special characters like brackets, use quotes in YAML:
      backend.megatron: ["llama1_8b_qkln", "llama1_8b_qkln"]
    - This naturally escapes the brackets for Hydra processing
    """
    from oellm_autoexp.orchestrator import ConfigSetup

    # Set up a Hydra-style config directory structure
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Create backend group directory
    backend_dir = config_dir / "backend"
    backend_dir.mkdir()

    # Create different backend variants
    base_backend = """
base_command: ["echo", "base"]
extra_cli_args: []
env: {}
"""
    variant1_backend = """
base_command: ["echo", "variant1"]
extra_cli_args: ["--variant1"]
env: {}
"""
    variant2_backend = """
base_command: ["echo", "variant2"]
extra_cli_args: ["--variant2", "--extra"]
env: {}
"""

    (backend_dir / "base.yaml").write_text(base_backend)
    (backend_dir / "variant1.yaml").write_text(variant1_backend)
    (backend_dir / "variant2.yaml").write_text(variant2_backend)

    # Create main config that uses defaults
    template_path = tmp_path / "template.sbatch"
    script_dir = tmp_path / "scripts"
    log_dir = tmp_path / "logs"
    base_output = tmp_path / "outputs"

    template_path.write_text("#!/bin/bash\n{sbatch_directives}\n\nsrun {srun_opts}{launcher_cmd}\n")

    main_config = f"""
defaults:
  - backend: base
  - _self_

project:
  name: hydra_test_${{index}}
  base_output_dir: {base_output}
  log_path: {log_dir}/slurm-%j.out
  log_path_current: {log_dir}/current.log

sweep:
  grids:
    - backend: [variant1, variant2]

slurm:
  template_path: {template_path}
  script_dir: {script_dir}
  log_dir: {log_dir}
  client:
    class_name: FakeSlurmClient

monitoring:
  class_name: NullMonitor
  log_path: {log_dir}/current.log

backend:
  class_name: NullBackend
  base_command: ["echo", "Hello"]

index: 0
"""

    config_path = config_dir / "test.yaml"
    config_path.write_text(main_config)

    # Build execution plan with ConfigSetup to enable Hydra override behavior
    from oellm_autoexp.config.loader import load_config_reference

    root = load_config_reference("test", config_dir, [])
    plan = build_execution_plan(
        root,
        config_setup=ConfigSetup(
            pwd=str(tmp_path),
            config_ref="test",
            config_dir=str(config_dir),
            override=[],
        ),
    )

    # Verify we have 2 jobs (one for each backend variant)
    assert len(plan.jobs) == 2

    # Render scripts to trigger the Hydra override application
    artifacts = render_scripts(plan)

    # Verify the artifacts were created
    assert len(artifacts.job_scripts) == 2
    assert len(artifacts.sweep_entries) == 2

    # Check that the launch commands reflect the different backend variants
    entry1 = artifacts.sweep_entries[0]
    entry2 = artifacts.sweep_entries[1]

    # Verify the launch commands are different and contain the expected variants
    argv1 = entry1["launch"]["argv"]
    argv2 = entry2["launch"]["argv"]

    assert argv1 != argv2, "Launch commands should be different for different backend variants"

    # Check that variant1 and variant2 commands are present
    variant1_found = any("variant1" in " ".join(argv) for argv in [argv1, argv2])
    variant2_found = any("variant2" in " ".join(argv) for argv in [argv1, argv2])

    assert variant1_found, "Should find variant1 in one of the launch commands"
    assert variant2_found, "Should find variant2 in one of the launch commands"

    # Verify that the extra_cli_args are applied correctly
    # variant2 should have both --variant2 and --extra
    variant2_argv = argv2 if "variant2" in " ".join(argv2) else argv1
    assert "--variant2" in variant2_argv, "variant2 should have --variant2 flag"
    assert "--extra" in variant2_argv, "variant2 should have --extra flag"


def test_hydra_multigroup_overrides_in_sweep(tmp_path: Path) -> None:
    """Test that sweep parameters can use Hydra group overrides.

    This test demonstrates that job.parameters like "backend=variant1" are applied
    as Hydra overrides, allowing sweeps over:
    - Hydra group selections (e.g., backend: [variant1, variant2])
    - Nested config values (e.g., backend.megatron.lr: [0.1, 0.01])
    - Any other configuration parameter

    Note on escaping:
    - For values with special characters like brackets, use quotes in YAML:
      backend.megatron: ["llama1_8b_qkln", "llama1_8b_qkln"]
    - This naturally escapes the brackets for Hydra processing
    """
    from oellm_autoexp.orchestrator import ConfigSetup

    # Set up a Hydra-style config directory structure
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Create backend group directory
    backend_dir = config_dir / "backend"
    backend_dir.mkdir()

    # Create different backend variants
    base_backend = """
base_command: ["echo", "base"]
extra_cli_args: []
env: {}
"""
    variant1_backend = """
base_command: ["echo", "variant1"]
env: {"variant1_was_here": "1"}
"""
    variant2_backend = """
base_command: ["echo", "variant2"]
extra_cli_args: ["--variant2", "--extra"]
env: {}
"""

    (backend_dir / "base.yaml").write_text(base_backend)
    (backend_dir / "variant1.yaml").write_text(variant1_backend)
    (backend_dir / "variant2.yaml").write_text(variant2_backend)

    # Create main config that uses defaults
    template_path = tmp_path / "template.sbatch"
    script_dir = tmp_path / "scripts"
    log_dir = tmp_path / "logs"
    base_output = tmp_path / "outputs"

    template_path.write_text("#!/bin/bash\n{sbatch_directives}\n\nsrun {srun_opts}{launcher_cmd}\n")

    main_config = f"""
defaults:
  - backend: base
  - _self_

project:
  name: hydra_test_${{index}}
  base_output_dir: {base_output}
  log_path: {log_dir}/slurm-%j.out
  log_path_current: {log_dir}/current.log

sweep:
  grids:
    - backend: ["variant1", "[variant1,variant2]"]

slurm:
  template_path: {template_path}
  script_dir: {script_dir}
  log_dir: {log_dir}
  client:
    class_name: FakeSlurmClient

monitoring:
  class_name: NullMonitor
  log_path: {log_dir}/current.log

backend:
  class_name: NullBackend
  base_command: ["echo", "Hello"]

index: 0
"""

    config_path = config_dir / "test.yaml"
    config_path.write_text(main_config)

    # Build execution plan with ConfigSetup to enable Hydra override behavior
    from oellm_autoexp.config.loader import load_config_reference

    root = load_config_reference("test", config_dir, [])
    plan = build_execution_plan(
        root,
        config_setup=ConfigSetup(
            pwd=str(tmp_path),
            config_ref="test",
            config_dir=str(config_dir),
            override=[],
        ),
    )

    # Verify we have 2 jobs (one for each backend variant)
    assert len(plan.jobs) == 2

    # Render scripts to trigger the Hydra override application
    artifacts = render_scripts(plan)

    # Verify the artifacts were created
    assert len(artifacts.job_scripts) == 2
    assert len(artifacts.sweep_entries) == 2

    # Check that the launch commands reflect the different backend variants
    entry1 = artifacts.sweep_entries[0]
    entry2 = artifacts.sweep_entries[1]

    # Verify the launch commands are different and contain the expected variants
    argv1 = entry1["launch"]["argv"]
    argv2 = entry2["launch"]["argv"]
    env1 = entry1["launch"]["env"]
    env2 = entry2["launch"]["env"]

    assert argv1 != argv2, "Launch commands should be different for different backend variants"

    # Check that variant1 and variant2 commands are present
    variant1_found = any("variant1_was_here" in env for env in [env1, env2])
    variant2_found = any("variant2" in " ".join(argv) for argv in [argv1, argv2])

    assert variant1_found, "Should find variant1 in one of the launch commands"
    assert variant2_found, "Should find variant2 in one of the launch commands"

    # Verify that the extra_cli_args are applied correctly
    # variant2 should have both --variant2 and --extra
    variant2_argv = argv2 if "variant2" in " ".join(argv2) else argv1
    assert "--variant2" in variant2_argv, "variant2 should have --variant2 flag"
    assert "--extra" in variant2_argv, "variant2 should have --extra flag"

    variant1_entry = entry1 if "variant1" in " ".join(argv1) else entry2
    assert "variant1_was_here" in variant1_entry["base_config"]["backend"]["env"]
