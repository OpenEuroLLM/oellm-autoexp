"""Test list composition with the user's basic/ config group setup."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
import pytest

from oellm_autoexp.hydra_staged_sweep.config.schema import SweepConfig
from oellm_autoexp.hydra_staged_sweep.config.loader import load_hydra_config
from oellm_autoexp.hydra_staged_sweep.expander import expand_sweep
from compoconf import NonStrictDataclass


@dataclass(init=False)
class BasicTestConfig(NonStrictDataclass):
    """Configuration for basic config group test - accepts any fields."""

    # Required fields from StagedSweepRoot
    sweep: SweepConfig = field(default_factory=SweepConfig)
    stage: str = ""
    index: int | tuple[int] = 0
    sibling: dict[str, Any] = field(default_factory=dict)


def test_basic_config_expansion():
    """Test what the expander produces for the basic config.

    Expected: 4 points from (2 configs in group 1) * (2 values in group 2)
    - Point 1: test="a1", basic=[integrate_a1, integrate_b1]
    - Point 2: test="a1", basic=[integrate_a1, integrate_b2]
    - Point 3: test="a2", basic=[integrate_a2, integrate_b1]
    - Point 4: test="a2", basic=[integrate_a2, integrate_b2]
    """
    config_dir = Path(__file__).parent / "configs" / "defaults_test"

    # Load config
    config = load_hydra_config(
        "config",
        config_dir=config_dir,
        config_class=BasicTestConfig,
    )

    # Check sweep configuration
    assert config.sweep.type == "product"
    assert "basic" in config.sweep.list_composition
    assert len(config.sweep.groups) == 2

    # Expand the sweep
    points = expand_sweep(config.sweep)

    # Group 1 is "list" mode with 2 configs
    # Group 2 is "product" mode with 2 values for basic
    # Result: 2 * 2 = 4 points
    print("\n=== Expansion Results ===")
    print(f"Total points: {len(points)}")

    for i, point in enumerate(points):
        print(f"\nPoint {i}:")
        print(f"  test: {point.parameters.get('test')}")
        print(f"  basic: {point.parameters.get('basic')}")
        print(f"  basic type: {type(point.parameters.get('basic'))}")

    # Expected with list_composition (once fixed):
    # Each point should have basic as a FLAT list with 2 elements
    assert len(points) == 4, f"Expected 4 points, got {len(points)}"

    # Check expected structure (will fail until we fix the nested list issue)
    for i, point in enumerate(points):
        basic = point.parameters.get("basic")
        test_val = point.parameters.get("test")

        print(f"\nPoint {i}: test={test_val}, basic={basic}")

        # Should be a list
        assert isinstance(basic, list), f"basic should be a list, got {type(basic)}"

        # Should have 2 elements (one from each group)
        assert len(basic) == 2, f"basic should have 2 elements, got {len(basic)}: {basic}"

        # IMPORTANT: After fix, all elements should be strings, not nested lists
        # Currently fails: basic=['integrate_a1', ['integrate_b1']]
        # Should be: basic=['integrate_a1', 'integrate_b1']
        for elem in basic:
            assert isinstance(elem, str), f"Elements should be strings, got {type(elem)}: {elem}"


def test_basic_config_hydra_resolution():
    """Test that Hydra resolution works when using proper overrides (without
    ++)."""
    pytest.skip("Will be implemented after fixing nested lists and override prefix logic")


def test_what_command_lines_would_be_generated():
    """Show what command-line overrides would be generated (before Hydra
    rejects them).

    This helps understand what the sweep is trying to do.
    """
    config_dir = Path(__file__).parent / "configs" / "defaults_test"

    config = load_hydra_config(
        "config",
        config_dir=config_dir,
        config_class=BasicTestConfig,
    )

    points = expand_sweep(config.sweep)

    print("\n=== Expected Command-Line Overrides ===")
    print("(These would be generated, but Hydra will reject them)")
    print()

    for i, point in enumerate(points):
        basic_val = point.parameters.get("basic")
        test_val = point.parameters.get("test")

        if isinstance(basic_val, list):
            # Format as Hydra list syntax
            basic_str = f"[{','.join(basic_val)}]"
        else:
            basic_str = str(basic_val)

        print(f"Point {i}: ++test={test_val} ++basic={basic_str}")


def test_show_expected_vs_actual():
    """Document what you expect vs what actually happens."""
    config_dir = Path(__file__).parent / "configs" / "defaults_test"

    config = load_hydra_config(
        "config",
        config_dir=config_dir,
        config_class=BasicTestConfig,
    )

    points = expand_sweep(config.sweep)

    print("\n" + "=" * 60)
    print("ACTUAL BEHAVIOR (current - with nested lists)")
    print("=" * 60)
    print(f"Points generated: {len(points)}")

    for i, point in enumerate(points):
        test_val = point.parameters.get("test")
        basic_val = point.parameters.get("basic")
        print(f"  {i + 1}. test={test_val}, basic={basic_val}")
