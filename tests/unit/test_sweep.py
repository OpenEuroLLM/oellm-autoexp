from compoconf import parse_config

from pathlib import Path

from oellm_autoexp.config.schema import RootConfig, SweepConfig
from oellm_autoexp.sweep.expander import expand_sweep
from oellm_autoexp.sweep.planner import build_job_plans


def _basic_root() -> RootConfig:
    data = {
        "project": {"name": "demo", "base_output_dir": "./outputs"},
        "sweep": {"axes": {"lr": [0.1, 0.01], "model": ["small", "large"]}},
        "slurm": {
            "template_path": "template.sbatch",
            "script_dir": "./scripts",
            "log_dir": "./logs",
            "launcher_cmd": "",
            "srun_opts": "",
            "client": {"class_name": "FakeSlurmClient"},
        },
        "monitoring": {"class_name": "NullMonitor"},
        "backend": {"class_name": "NullBackend"},
    }
    return parse_config(RootConfig, data)


def test_expand_sweep_cartesian_product():
    sweep_cfg = SweepConfig(axes={"lr": [0.1, 0.01], "layers": [8, 16]})
    points = expand_sweep(sweep_cfg)
    values = [p.parameters for p in points]
    assert len(values) == 4
    assert values[0]["lr"] == 0.1
    assert values[0]["layers"] == 8


def test_build_job_plans_name_template():
    root = _basic_root()
    points = expand_sweep(root.sweep)
    jobs = build_job_plans(root, points)
    assert len(jobs) == 4
    assert jobs[0].name.startswith("demo")
    assert Path(jobs[0].log_path).name == "slurm.out"


def test_build_job_plans_extracts_lifecycle_fields():
    root = _basic_root()
    root.monitoring.start_condition_cmd = None
    root.monitoring.termination_string = "all done"
    root.monitoring.inactivity_threshold_seconds = 123
    root.monitoring.start_condition_interval_seconds = 12
    root.sweep.axes = {
        "job.start_condition_cmd": ["echo 1"],
        "monitoring.termination_string": ["Finished"],
        "job.start_condition_interval_seconds": [30],
        "job.inactivity_threshold_seconds": [45],
        "lr": [0.1],
    }

    points = expand_sweep(root.sweep)
    jobs = build_job_plans(root, points)

    assert len(jobs) == 1
    job = jobs[0]
    assert job.start_condition_cmd == "echo 1"
    assert job.start_condition_interval_seconds == 30
    assert job.termination_string == "Finished"
    assert job.inactivity_threshold_seconds == 45
    assert "job.start_condition_cmd" not in job.parameters
    assert "monitoring.termination_string" not in job.parameters
    assert job.parameters["lr"] == "0.1"


def test_expand_sweep_with_base_values():
    sweep_cfg = SweepConfig(axes=None, base_values={"foo": "bar"})
    points = expand_sweep(sweep_cfg)
    assert len(points) == 1
    assert points[0].parameters["foo"] == "bar"


def test_expand_sweep_scalars_and_lists():
    sweep_cfg = SweepConfig(axes={"flags": ["a", "b", "c"], "nested": [{"x": [1, 2]}]})
    points = expand_sweep(sweep_cfg)
    values = [p.parameters for p in points]
    assert len(values) == 6
    assert any(v["flags"] == "a" for v in values)
    assert any(v["nested.x"] == 2 for v in values)


def test_expand_sweep_filter_expression():
    # Only combinations where a * b <= 40 are kept
    sweep_cfg = SweepConfig(
        axes={"a": [1, 2, 3], "b": [10, 20, 30]},
        filter="a * b <= 40",
    )
    points = expand_sweep(sweep_cfg)
    values = [p.parameters for p in points]
    assert len(values) == 4
    assert all(v["a"] * v["b"] <= 40 for v in values)
