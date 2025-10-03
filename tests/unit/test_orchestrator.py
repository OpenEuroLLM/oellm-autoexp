from pathlib import Path

from oellm_autoexp.orchestrator import build_execution_plan, render_scripts


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
    scripts = render_scripts(plan)

    assert len(plan.jobs) == 2
    assert len(scripts) == 2
    for script in scripts:
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
  base_output_dir: {tmp_path / 'outputs'}
  environment:
    ENV: 1
sweep:
  axes:
    lr: [0.1]
slurm:
  template_path: {template_path}
  script_dir: {script_dir}
  log_dir: {log_dir}
  launcher_cmd: "module load foo &&"
  srun_opts: "--ntasks=4"
  sbatch_overrides:
    nodes: 2
  sbatch_extra_directives:
    - "#SBATCH --constraint=volta"
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
    scripts = render_scripts(plan)

    script_path = scripts[0]
    content = script_path.read_text()
    assert "#SBATCH --nodes=2" in content
    assert "#SBATCH --constraint=volta" in content
    assert "export ENV=1" in content
    assert "srun --ntasks=4 module load foo &&" in content
