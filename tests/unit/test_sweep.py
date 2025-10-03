from compoconf import parse_config

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
        },
        "monitoring": {"implementation": {"class_name": "NullMonitor"}},
        "backend": {"implementation": {"class_name": "NullBackend"}},
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
    assert jobs[0].log_path.name == "slurm.out"


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
