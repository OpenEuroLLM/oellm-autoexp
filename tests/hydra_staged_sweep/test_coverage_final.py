"""Targeted tests to achieve 100% coverage for remaining uncovered lines."""

import pytest
import tempfile
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from compoconf import ConfigInterface
from oellm_autoexp.hydra_staged_sweep.config.loader import (
    load_config,
    load_config_reference,
)
from oellm_autoexp.hydra_staged_sweep.config.schema import StagedSweepRoot
from oellm_autoexp.hydra_staged_sweep.dag_resolver import (
    find_sibling_by_group_path,
)
from oellm_autoexp.hydra_staged_sweep.expander import SweepPoint


def test_config_reference_json_exception(caplog):
    """Cover lines 242-243: exception when loading config_reference.json.

    This test triggers the fallback path when config_reference.json exists
    but cannot be loaded due to invalid JSON.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text("sweep:\n  type: product\n")

        # Create invalid JSON that will trigger exception at line 229-242
        ref_json = Path(tmpdir) / "config_reference.json"
        ref_json.write_text("{invalid json")

        # Should catch exception and print warning (lines 242-243)
        # Pass overrides to ensure we go through that path
        result = load_config_reference(
            config_path=str(config_path),
            overrides=["sweep.type=list"],
            config_class=StagedSweepRoot,
        )
        assert result is not None

    # Verify the warning was logged
    assert "Could not load config_reference.json" in caplog.text


def test_parse_config_exception():
    """Cover lines 265-266: exception when parse_config fails in load_config_reference.

    This test triggers the exception handler when parse_config cannot
    parse the configuration data.
    """

    @dataclass(kw_only=True)
    class StrictConfig(ConfigInterface):
        required_field: str  # Required field with no default

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        # Missing required_field will cause parse_config to fail
        config_path.write_text("other_field: value\n")

        # Test in load_config_reference (lines 265-266)
        with pytest.raises(
            ValueError,
            match=r"Undefined keys {'other_field'} and unset keys {'required_field'} in data",
        ):
            load_config_reference(
                config_path=str(config_path),
                overrides=["other_field=changed"],  # Pass overrides to go through that path
                config_class=StrictConfig,
            )


def test_metadata_setdefault_coverage():
    """Cover lines 269-270: metadata.setdefault calls.

    This test ensures that when a config class has metadata attribute,
    the setdefault calls are executed.
    """

    # Create a config class with metadata attribute
    @dataclass(kw_only=True)
    class ConfigWithMetadata(StagedSweepRoot):
        metadata: dict[str, Any] = field(default_factory=dict)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text("sweep:\n  type: product\nmetadata: {}\n")

        # Pass overrides to go through the metadata path
        result = load_config_reference(
            config_path=str(config_path),
            overrides=["sweep.type=list"],
            config_class=ConfigWithMetadata,
        )

        # Lines 269-270 should have set these
        assert hasattr(result, "metadata")
        assert "config_ref" in result.metadata
        assert "config_dir" in result.metadata
        assert str(config_path) in result.metadata["config_ref"]


def test_multiple_siblings_warning_log(caplog):
    """Cover line 102: warning when multiple siblings match.

    This test creates a scenario where multiple siblings have the same
    stage name, triggering the warning at line 102.
    """
    # Set logging level to capture warnings for the module
    with caplog.at_level(logging.WARNING):
        # Create points where multiple siblings match the pattern
        # p0 MUST have a sibling reference to trigger the find_sibling logic
        p0 = SweepPoint(
            index=0,
            parameters={"stage": "base", "ref": "${sibling.train.x}"},  # Has sibling reference
            group_path=(0, 0),
            stage_path=(False, True),
        )
        # Both p1 and p2 have stage="train" - this creates multiple matches
        # They need similar group_paths so both are considered siblings of p0
        p1 = SweepPoint(
            index=1,
            parameters={"stage": "train", "x": 1},
            group_path=(0, 1),
            stage_path=(False, True),
        )
        p2 = SweepPoint(
            index=2,
            parameters={"stage": "train", "x": 2},  # Duplicate stage name
            group_path=(0, 2),  # Changed to (0, 2) so it's also a sibling of p0
            stage_path=(False, True),
        )

        points = {0: p0, 1: p1, 2: p2}

        # This should trigger line 102 warning because both p1 and p2 match "train"
        result = find_sibling_by_group_path(p0, points, "train")

        # Verify warning was logged
        assert result is not None
        assert any("Multiple matched siblings" in str(record.msg) for record in caplog.records)


def test_build_dag_valueerror_exception():
    """Cover lines 127-128: except ValueError in build_dependency_dag."""
    from unittest.mock import patch  # noqa
    from oellm_autoexp.hydra_staged_sweep.dag_resolver import build_dependency_dag_from_points

    with patch(
        "oellm_autoexp.hydra_staged_sweep.dag_resolver.find_sibling_by_group_path"
    ) as mock_find:
        mock_find.side_effect = ValueError("Test error")

        p0 = SweepPoint(
            index=0,
            parameters={"ref": "${sibling.x.y}"},
            group_path=(0,),
            stage_path=(False,),
        )

        points = {0: p0}

        # Should catch ValueError (lines 127-128)
        dag = build_dependency_dag_from_points(points)

        assert dag.number_of_nodes() == 1


def test_resolve_sweep_dict_input():
    """Cover line 211: else branch when points is already dict."""
    from oellm_autoexp.hydra_staged_sweep.dag_resolver import resolve_sweep_with_dag

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text("sweep:\n  type: product")

        config = load_config(str(config_path))
        from oellm_autoexp.hydra_staged_sweep.config.schema import ConfigSetup

        setup = ConfigSetup(pwd=".", config_path=str(config_path), config_dir=tmpdir)

        # Pass points as dict (line 211 is the else branch)
        points_dict = {0: SweepPoint(index=0, parameters={}, group_path=(0,), stage_path=(False,))}

        plans = resolve_sweep_with_dag(config, points_dict, setup)
        assert len(plans) == 1


def test_unknown_group_type():
    """Cover line 246: raise ValueError for unknown group type."""
    from oellm_autoexp.hydra_staged_sweep.expander import _expand_group

    with pytest.raises(ValueError, match="Unknown group type"):
        _expand_group(
            group_type="invalid_type",
            groups=[{"params": {"x": [1]}}],
            base_values={},
            group_path=(),
            stage_path=(),
            list_composition=set(),
        )


def test_cartesian_product_empty():
    """Cover line 257: empty list returns empty."""
    from oellm_autoexp.hydra_staged_sweep.expander import _cartesian_product_groups

    result = _cartesian_product_groups([], set())
    assert result == []
