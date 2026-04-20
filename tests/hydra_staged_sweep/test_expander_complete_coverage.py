"""Comprehensive tests for expander.py to achieve 100% coverage."""

from hydra_staged_sweep.expander import expand_sweep, _product_dict
from hydra_staged_sweep.config.schema import SweepConfig
from omegaconf import DictConfig


def test_expand_group_with_empty_groups():
    """Test _expand_group with empty groups list."""
    from hydra_staged_sweep.expander import _expand_group

    result = _expand_group(
        group_type="product",
        groups=[],
        base_values={"param": "value"},
        group_path=(),
        stage_path=(),
        list_composition=set(),
    )

    # Should return the base values
    assert len(result) == 1
    assert result[0][0] == {"param": "value"}


def test_expand_sweep_with_stage_in_params():
    """Test expansion when 'stage' is in parameters."""
    config = SweepConfig(
        type="product",
        groups=[
            {
                "type": "product",
                "params": {
                    "stage": ["train", "eval"],
                    "value": [1, 2],
                },
            }
        ],
    )

    points = expand_sweep(config)

    # Should have 4 combinations (2 stages x 2 values)
    assert len(points) == 4

    # Check that stage_path is properly set
    for point in points:
        # stage_path should have True somewhere since 'stage' is present
        assert any(point.stage_path)


def test_expand_sweep_with_nested_groups_containing_stage():
    """Test expansion with nested groups that contain 'stage' parameter."""
    config = SweepConfig(
        type="product",
        groups=[
            {
                "type": "list",
                "configs": [
                    {
                        "type": "product",
                        "params": {
                            "stage": ["stage1", "stage2"],
                        },
                    }
                ],
            }
        ],
    )

    points = expand_sweep(config)

    # Should have 2 points (one for each stage)
    assert len(points) == 2


def test_expand_sweep_with_list_configs_and_nested_groups():
    """Test expansion with list configs containing nested group structures."""
    config = SweepConfig(
        type="product",
        groups=[
            {
                "type": "list",
                "configs": [
                    {
                        "type": "product",
                        "groups": [
                            {
                                "type": "product",
                                "params": {"stage": ["A", "B"]},
                            }
                        ],
                    },
                    {"stage": "C", "param": "value"},
                ],
            }
        ],
    )

    points = expand_sweep(config)

    # Should have 3 points: A, B, C
    assert len(points) == 3
    stages = [p.parameters.get("stage") for p in points]
    assert set(stages) == {"A", "B", "C"}


def test_expand_sweep_with_defaults_in_group():
    """Test expansion with defaults at group level."""
    config = SweepConfig(
        type="product",
        groups=[
            {
                "type": "product",
                "defaults": {"default_param": "default_value"},
                "params": {
                    "learning_rate": [0.001, 0.01],
                },
            }
        ],
    )

    points = expand_sweep(config)

    # All points should have the default_param
    assert len(points) == 2
    for point in points:
        assert point.parameters["default_param"] == "default_value"


def test_product_dict_with_empty_config():
    """Test _product_dict with empty DictConfig."""
    cfg = DictConfig({})
    result = _product_dict(cfg)

    assert result == [{}]


def test_expand_sweep_with_nested_list_in_configs():
    """Test expansion with nested configs that contain stage."""
    config = SweepConfig(
        type="product",
        groups=[
            {
                "type": "list",
                "configs": [
                    {
                        "configs": [{"stage": "nested_A"}],
                    }
                ],
            }
        ],
    )

    points = expand_sweep(config)

    assert len(points) >= 1
    # Verify that nested stages are properly handled


def test_expand_sweep_with_simple_config_in_list():
    """Test expansion where list configs contain simple dicts without stage."""
    config = SweepConfig(
        type="product",
        groups=[
            {
                "type": "list",
                "configs": [
                    {"param1": "A", "param2": 1},
                    {"param1": "B", "param2": 2},
                ],
            }
        ],
    )

    points = expand_sweep(config)

    assert len(points) == 2
    assert points[0].parameters["param1"] == "A"
    assert points[1].parameters["param1"] == "B"
