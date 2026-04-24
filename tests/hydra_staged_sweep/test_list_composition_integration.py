"""Integration tests for list composition feature."""

import tempfile
from pathlib import Path
from dataclasses import dataclass, field

from oellm_autoexp.hydra_staged_sweep.config.schema import SweepConfig, StagedSweepRoot, ConfigSetup
from oellm_autoexp.hydra_staged_sweep.expander import expand_sweep
from oellm_autoexp.hydra_staged_sweep.dag_resolver import resolve_sweep_with_dag


@dataclass(kw_only=True)
class TestConfig(StagedSweepRoot):
    """Test configuration with plugins."""

    plugins: list[str] | str = field(default_factory=list)
    model: str = "base"
    learning_rate: float = 0.001


def test_list_composition_end_to_end():
    """Test list composition from sweep expansion through command-line
    generation."""
    config = TestConfig(
        sweep=SweepConfig(
            type="product",
            groups=[
                {"params": {"plugins": ["logger", "wandb"]}},  # 2 values
                {
                    "params": {"plugins": ["tensorboard"], "learning_rate": [0.001, 0.01]}
                },  # 1 * 2 = 2
            ],
            list_composition=["plugins"],
        )
    )

    # Expand sweep
    points = expand_sweep(config.sweep)
    # Product: 2 (plugins from group 1) * 2 (learning_rate from group 2) = 4
    assert len(points) == 4

    # Check that plugins are accumulated from both groups
    for point in points:
        assert isinstance(point.parameters["plugins"], list)
        assert len(point.parameters["plugins"]) == 2  # One from each group
        # First element is from group 1 (logger or wandb)
        assert point.parameters["plugins"][0] in ["logger", "wandb"]
        # Second element is from group 2
        assert point.parameters["plugins"][1] == "tensorboard"

    # Check specific combinations exist
    assert any(
        p.parameters["plugins"] == ["logger", "tensorboard"]
        and p.parameters["learning_rate"] == 0.001
        for p in points
    )
    assert any(
        p.parameters["plugins"] == ["wandb", "tensorboard"]
        and p.parameters["learning_rate"] == 0.01
        for p in points
    )


def test_list_composition_with_dag_resolver():
    """Test that list composition works through the full DAG resolution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_content = """
plugins: []
model: base

sweep:
  type: product
  list_composition:
    - plugins
  groups:
    - params:
        plugins: ["a", "b"]  # 2 values
    - params:
        plugins: ["c"]
        model: ["small", "large"]  # 2 values
"""
        config_path.write_text(config_content)

        # Load and resolve
        from oellm_autoexp.hydra_staged_sweep.config.loader import load_config

        config = load_config(config_path, config_class=TestConfig)
        points = expand_sweep(config.sweep)

        # Product: 2 (plugins) * 2 (model) = 4 points
        assert len(points) == 4

        # Resolve with DAG
        setup = ConfigSetup(pwd=str(tmpdir), config_path=str(config_path))
        plans = resolve_sweep_with_dag(config, points, setup, config_class=TestConfig)

        assert len(plans) == 4

        # Check command-line parameters contain list syntax
        for plan in plans:
            # Find the plugins parameter in the command line
            plugins_params = [p for p in plan.parameters if "plugins=" in p]
            assert len(plugins_params) > 0

            # Should use the [x,c] format for string lists (one from each group)
            plugins_param = plugins_params[0]
            # Each point has a plugin from group 1 (a or b) and one from group 2 (c)
            assert "plugins=[a,c]" in plugins_param or "plugins=[b,c]" in plugins_param


def test_list_composition_multiple_params():
    """Test list composition with multiple parameters being accumulated."""
    config = TestConfig(
        sweep=SweepConfig(
            type="product",
            groups=[
                {"params": {"plugins": ["a"], "model": ["m1"]}},
                {"params": {"plugins": ["b"], "model": ["m2"]}},
            ],
            list_composition=["plugins", "model"],
        )
    )

    points = expand_sweep(config.sweep)
    assert len(points) == 1

    # Both plugins and model should be accumulated
    assert points[0].parameters["plugins"] == ["a", "b"]
    assert points[0].parameters["model"] == ["m1", "m2"]


def test_list_composition_partial():
    """Test that only specified parameters are accumulated, others are
    overridden."""
    config = TestConfig(
        sweep=SweepConfig(
            type="product",
            groups=[
                {"params": {"plugins": ["a"], "learning_rate": [0.001]}},
                {"params": {"plugins": ["b"], "learning_rate": [0.01]}},
            ],
            list_composition=["plugins"],  # Only plugins, not learning_rate
        )
    )

    points = expand_sweep(config.sweep)
    assert len(points) == 1

    # plugins accumulated
    assert points[0].parameters["plugins"] == ["a", "b"]
    # learning_rate overridden (last value wins)
    assert points[0].parameters["learning_rate"] == 0.01
