from compoconf import parse_config

from pathlib import Path

from oellm_autoexp.config.schema import RootConfig, SweepConfig
from oellm_autoexp.sweep.expander import expand_sweep
from oellm_autoexp.sweep.planner import build_job_plans

# Import to register configs in registry
import oellm_autoexp.monitor.watcher  # noqa: F401
import oellm_autoexp.backends.base  # noqa: F401
import oellm_autoexp.slurm.client  # noqa: F401


def _basic_root() -> RootConfig:
    data = {
        "project": {"name": "demo", "base_output_dir": "./outputs"},
        "sweep": {"grids": [{"backend.megatron.lr": [0.1, 0.01]}]},
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
    sweep_cfg = SweepConfig(
        grids=[{"backend.megatron.lr": [0.1, 0.01], "backend.megatron.num_layers": [8, 16]}]
    )
    points = expand_sweep(sweep_cfg)
    values = [p.parameters for p in points]
    assert len(values) == 4
    assert values[0]["backend.megatron.lr"] == 0.1
    assert values[0]["backend.megatron.num_layers"] == 8


def test_build_job_plans_name_template():
    root = _basic_root()
    points = expand_sweep(root.sweep)
    jobs = build_job_plans(root, points)
    assert len(jobs) == 2
    assert jobs[0].name.startswith("demo")
    assert Path(jobs[0].log_path).name == "slurm-%j.out"


def test_build_job_plans_extracts_lifecycle_fields():
    root = _basic_root()
    root.monitoring.start_condition_cmd = None
    root.monitoring.termination_string = "all done"
    root.monitoring.inactivity_threshold_seconds = 123
    root.monitoring.start_condition_interval_seconds = 12
    root.sweep.grids = [
        {
            "job.start_condition_cmd": ["echo 1"],
            "monitoring.termination_string": ["Finished"],
            "job.start_condition_interval_seconds": [30],
            "job.inactivity_threshold_seconds": [45],
            "lr": [0.1],
        }
    ]

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


def test_build_job_plans_extracts_start_conditions():
    """Test that start_conditions (list-based) are properly extracted."""
    root = _basic_root()
    root.sweep.grids = [
        {
            "job.start_conditions": [
                [
                    {
                        "class_name": "FileExistsCondition",
                        "path": "/checkpoint/done.txt",
                        "blocking": True,
                    },
                    {
                        "class_name": "MetadataCondition",
                        "key": "checkpoint_iteration",
                        "equals": "80000",
                    },
                ]
            ],
            "lr": [0.1],
        }
    ]

    points = expand_sweep(root.sweep)
    jobs = build_job_plans(root, points)

    assert len(jobs) == 1
    job = jobs[0]
    assert len(job.start_conditions) == 2
    assert job.start_conditions[0]["class_name"] == "FileExistsCondition"
    assert job.start_conditions[0]["path"] == "/checkpoint/done.txt"
    assert job.start_conditions[1]["class_name"] == "MetadataCondition"
    assert "job.start_conditions" not in job.parameters
    assert job.parameters["lr"] == "0.1"


def test_expand_sweep_with_base_values():
    sweep_cfg = SweepConfig(grids=[], base_values={"foo": "bar"})
    points = expand_sweep(sweep_cfg)
    assert len(points) == 1
    assert points[0].parameters["foo"] == "bar"


def test_expand_sweep_scalars_and_lists():
    sweep_cfg = SweepConfig(grids=[{"flags": ["a", "b", "c"], "nested.x": [1, 2]}])
    points = expand_sweep(sweep_cfg)
    values = [p.parameters for p in points]
    assert len(values) == 6
    assert any(v["flags"] == "a" for v in values)
    assert any(v["nested.x"] == 2 for v in values)


def test_expand_sweep_filter_expression():
    # Only combinations where a * b <= 40 are kept
    sweep_cfg = SweepConfig(
        grids=[{"a": [1, 2, 3], "b": [10, 20, 30]}],
        filter="a * b <= 40",
    )
    points = expand_sweep(sweep_cfg)
    values = [p.parameters for p in points]
    assert len(values) == 6
    assert all(v["a"] * v["b"] <= 40 for v in values)


def test_expand_composable_sweep_product_mode():
    """Test composable sweep with product mode (cartesian product)."""
    sweep_cfg = SweepConfig(
        type="product",
        groups=[
            {
                "type": "product",
                "params": {
                    "backend.megatron.lr": [1e-4, 5e-4],
                    "backend.megatron.global_batch_size": [64, 128],
                },
            },
        ],
    )
    points = expand_sweep(sweep_cfg)
    assert len(points) == 4  # 2 lr * 2 batch_size
    values = [p.parameters for p in points]
    assert any(
        v["backend.megatron.lr"] == 1e-4 and v["backend.megatron.global_batch_size"] == 64
        for v in values
    )


def test_expand_composable_sweep_list_mode():
    """Test composable sweep with list mode (no cross-product)."""
    sweep_cfg = SweepConfig(
        type="list",
        groups=[
            {
                "type": "list",
                "configs": [
                    {"stage": "stable", "backend.megatron.lr_wsd_decay_iters": 0},
                    {"stage": "cooldown", "backend.megatron.lr_wsd_decay_iters": 1000},
                ],
            },
        ],
    )
    points = expand_sweep(sweep_cfg)
    assert len(points) == 2  # No cross-product, just 2 configs
    values = [p.parameters for p in points]
    assert any(v["stage"] == "stable" for v in values)
    assert any(v["stage"] == "cooldown" for v in values)


def test_expand_composable_sweep_mixed_product_list():
    """Test composable sweep mixing product and list groups."""
    sweep_cfg = SweepConfig(
        type="product",
        groups=[
            {
                "type": "product",
                "params": {"backend.megatron.lr": [1e-4, 5e-4]},
            },
            {
                "type": "list",
                "configs": [
                    {"stage": "stable"},
                    {"stage": "cooldown"},
                ],
            },
        ],
    )
    points = expand_sweep(sweep_cfg)
    assert len(points) == 4  # 2 lr * 2 stages (product of groups)
    values = [p.parameters for p in points]
    # Should have all combinations of lr and stage
    assert any(v["backend.megatron.lr"] == 1e-4 and v["stage"] == "stable" for v in values)
    assert any(v["backend.megatron.lr"] == 1e-4 and v["stage"] == "cooldown" for v in values)
    assert any(v["backend.megatron.lr"] == 5e-4 and v["stage"] == "stable" for v in values)
    assert any(v["backend.megatron.lr"] == 5e-4 and v["stage"] == "cooldown" for v in values)


def test_expand_composable_sweep_with_filter():
    """Test composable sweep with filter expression."""
    sweep_cfg = SweepConfig(
        type="product",
        groups=[
            {
                "type": "product",
                "params": {"a": [1, 2, 3], "b": [10, 20, 30]},
            },
        ],
        filter="a * b <= 40",
    )
    points = expand_sweep(sweep_cfg)
    values = [p.parameters for p in points]
    assert len(values) == 6
    assert all(v["a"] * v["b"] <= 40 for v in values)


def test_expand_nested_product_in_list():
    """Test nested product groups inside a list group (union of grids)."""
    sweep_cfg = SweepConfig(
        type="product",
        groups=[
            # List of two product grids (union)
            {
                "type": "list",
                "configs": [
                    {
                        "type": "product",
                        "params": {"lr": [1e-4, 5e-4], "batch": [64, 128]},
                    },  # 4 configs
                    {
                        "type": "product",
                        "params": {"lr": [1e-3, 2e-3], "batch": [256]},
                    },  # 2 configs
                ],
            },
            # Stages
            {"type": "list", "configs": [{"stage": "stable"}, {"stage": "cooldown"}]},
        ],
    )
    points = expand_sweep(sweep_cfg)
    assert len(points) == 12  # (4 + 2) × 2 = 12

    # Verify we have union of grids
    lr_batch_combos = set()
    for p in points:
        lr = p.parameters.get("lr")
        batch = p.parameters.get("batch")
        lr_batch_combos.add((lr, batch))

    assert len(lr_batch_combos) == 6  # 4 from grid1 + 2 from grid2
    assert (1e-4, 64) in lr_batch_combos  # From grid1
    assert (1e-3, 256) in lr_batch_combos  # From grid2


def test_expand_with_group_defaults():
    """Test group-level defaults that get merged into all configs."""
    sweep_cfg = SweepConfig(
        type="list",
        groups=[
            {
                "type": "list",
                "defaults": {
                    "common_param": "shared_value",
                    "another_common": 42,
                },
                "configs": [
                    {"stage": "stage1", "specific_param": "value1"},
                    {"stage": "stage2", "specific_param": "value2"},
                    {"stage": "stage3", "specific_param": "value3", "common_param": "override"},
                ],
            },
        ],
    )
    points = expand_sweep(sweep_cfg)
    assert len(points) == 3

    # All configs should have defaults
    assert all(p.parameters["another_common"] == 42 for p in points)

    # First two should have default common_param
    assert points[0].parameters["common_param"] == "shared_value"
    assert points[1].parameters["common_param"] == "shared_value"

    # Third should override the default
    assert points[2].parameters["common_param"] == "override"

    # All should have their specific params
    assert points[0].parameters["specific_param"] == "value1"
    assert points[1].parameters["specific_param"] == "value2"
    assert points[2].parameters["specific_param"] == "value3"


def test_expand_with_nested_defaults():
    """Test defaults with nested dict values (like job.start_conditions)."""
    sweep_cfg = SweepConfig(
        type="list",
        groups=[
            {
                "type": "list",
                "defaults": {
                    "job.start_conditions": [
                        {"class_name": "FileExistsCondition", "path": "/shared/path"},
                    ],
                    "job.cancel_conditions": [
                        {"class_name": "SlurmStateCondition", "state": "FAILED"},
                    ],
                    "common_scalar": 100,
                },
                "configs": [
                    {"stage": "decay6B", "tokens": 6_000_000_000},
                    {"stage": "decay12B", "tokens": 12_000_000_000},
                ],
            },
        ],
    )
    points = expand_sweep(sweep_cfg)
    assert len(points) == 2

    # All configs should have the defaults
    for p in points:
        assert p.parameters["common_scalar"] == 100
        assert isinstance(p.parameters["job.start_conditions"], list)
        assert len(p.parameters["job.start_conditions"]) == 1
        assert p.parameters["job.start_conditions"][0]["class_name"] == "FileExistsCondition"
        assert isinstance(p.parameters["job.cancel_conditions"], list)

    # Each should have its specific tokens
    assert points[0].parameters["tokens"] == 6_000_000_000
    assert points[1].parameters["tokens"] == 12_000_000_000


def test_expand_with_product_group_defaults():
    """Test defaults work with product groups too."""
    sweep_cfg = SweepConfig(
        type="product",
        groups=[
            {
                "type": "product",
                "defaults": {
                    "model_type": "dense",
                    "architecture": "transformer",
                },
                "params": {
                    "lr": [1e-4, 5e-4],
                    "batch_size": [64, 128],
                },
            },
        ],
    )
    points = expand_sweep(sweep_cfg)
    assert len(points) == 4  # 2 lr × 2 batch_size

    # All should have defaults
    for p in points:
        assert p.parameters["model_type"] == "dense"
        assert p.parameters["architecture"] == "transformer"

    # All should have their specific lr and batch_size
    assert any(p.parameters["lr"] == 1e-4 and p.parameters["batch_size"] == 64 for p in points)
    assert any(p.parameters["lr"] == 5e-4 and p.parameters["batch_size"] == 128 for p in points)
