"""Test edge cases for complete coverage."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


from hydra_staged_sweep.expander import expand_sweep, SweepPoint
from hydra_staged_sweep.dag_resolver import resolve_sweep_with_dag
from hydra_staged_sweep.config.schema import ConfigSetup, SweepConfig
from hydra_staged_sweep.config.loader import load_hydra_config
from compoconf import NonStrictDataclass


def test_expand_sweep_with_none_config():
    """Test expand_sweep when config is None."""
    points = expand_sweep(None)

    assert len(points) == 1
    assert points[0].index == 0
    assert points[0].parameters == []


@dataclass(init=False)
class EdgeCaseTestConfig(NonStrictDataclass):
    """Test config for edge cases."""

    sweep: SweepConfig = field(default_factory=SweepConfig)
    stage: str = ""
    index: int | tuple[int] = 0
    sibling: dict[str, Any] = field(default_factory=dict)


def test_resolve_sweep_with_none_sweep():
    """Test resolve_sweep_with_dag when config.sweep is None."""
    config_dir = Path(__file__).parent / "configs" / "defaults_test"

    # Create a config with no sweep
    config = EdgeCaseTestConfig()
    config.sweep = None

    # Create a single point with empty parameters
    points = [SweepPoint(index=0, parameters={})]

    setup = ConfigSetup(
        pwd=str(config_dir),
        config_name="config",
        config_dir=str(config_dir),
    )

    # This should handle the None sweep case
    plans = resolve_sweep_with_dag(config, points, setup, config_class=EdgeCaseTestConfig)

    assert len(plans) == 1
    assert plans[0].sibling_pattern is None


def test_config_group_detected_with_actual_directory():
    """Integration test ensuring config group detection works in full
    resolution."""
    config_dir = Path(__file__).parent / "configs" / "defaults_test"

    # Load config
    config = load_hydra_config("config", config_dir=config_dir, config_class=EdgeCaseTestConfig)

    # Expand sweep
    points = expand_sweep(config.sweep)

    # Resolve with DAG - this should use config group detection
    setup = ConfigSetup(
        pwd=str(config_dir),
        config_name="config",
        config_dir=str(config_dir),
    )

    plans = resolve_sweep_with_dag(config, points, setup, config_class=EdgeCaseTestConfig)

    # Verify that basic parameter was detected as config group
    assert len(plans) > 0

    # Check that basic parameter uses no prefix (config group)
    for plan in plans:
        basic_params = [p for p in plan.parameters if "basic=" in p]
        if basic_params:
            # Should be "basic=[...]" not "++basic=[...]"
            assert basic_params[0].startswith("basic=")
            assert not basic_params[0].startswith("++basic=")
