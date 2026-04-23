import pytest
from oellm_autoexp.hydra_staged_sweep.config.schema import SweepConfig
from oellm_autoexp.hydra_staged_sweep.expander import expand_sweep


def test_expand_empty_groups():
    config = SweepConfig(type="product", groups=[])
    points = expand_sweep(config)
    assert len(points) == 1
    assert points[0].index == 0


def test_expand_nested_groups_complex():
    # Test nested groups without 'params' or 'configs' but with 'groups'
    config = SweepConfig(
        type="list",
        groups=[{"type": "product", "groups": [{"params": {"a": [1]}}, {"params": {"b": [2]}}]}],
    )
    points = expand_sweep(config)
    assert len(points) == 1
    assert points[0].parameters["a"] == 1
    assert points[0].parameters["b"] == 2


def test_expand_group_error():
    config = SweepConfig(type="list", groups=[{"invalid": "key"}])
    with pytest.raises(ValueError, match="Group must have 'groups', 'params', or 'configs'"):
        expand_sweep(config)


def test_expand_no_groups_defined():
    config = SweepConfig(type=None, groups=None)
    points = expand_sweep(config)
    assert len(points) == 1
    assert points[0].index == 0


def test_expand_list_group_with_empty_configs():
    config = SweepConfig(type="list", groups=[{"configs": []}])
    # _expand_group returns [ (base_values, ...) ] if no configs?
    # Actually 'all_combinations' will have an empty list for this group if no configs.
    # Wait, if configs is empty, combinations remains empty.
    # If all groups are empty, result might be empty.
    points = expand_sweep(config)
    assert len(points) == 1


def test_cartesian_product_groups_logic():
    # Test _cartesian_product_groups with multiple groups
    config = SweepConfig(
        type="product", groups=[{"params": {"a": [1, 2]}}, {"params": {"b": [3, 4]}}]
    )
    points = expand_sweep(config)
    assert len(points) == 4
