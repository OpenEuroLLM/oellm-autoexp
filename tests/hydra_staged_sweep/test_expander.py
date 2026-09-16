from oellm_autoexp.hydra_staged_sweep.config.schema import SweepConfig
from oellm_autoexp.hydra_staged_sweep.expander import expand_sweep


def test_composable_product_group():
    config = SweepConfig(
        type="product",
        groups=[{"params": {"a": [1, 2]}}, {"params": {"b": [3, 4]}}],
        base_values={"base": 0},
    )
    points = expand_sweep(config)
    assert len(points) == 4  # 2 * 2

    # Check for a specific combination
    assert any(p.parameters["a"] == 1 and p.parameters["b"] == 3 for p in points)
    assert any(p.parameters["a"] == 2 and p.parameters["b"] == 4 for p in points)


def test_composable_list_group():
    config = SweepConfig(
        type="list",
        groups=[
            {"params": {"a": [1, 2]}},  # 2 items
            {"params": {"a": [3, 4]}},  # 2 items
        ],
        base_values={},
    )
    points = expand_sweep(config)
    assert len(points) == 4  # 2 + 2, concatenated

    values = sorted([p.parameters["a"] for p in points])
    assert values == [1, 2, 3, 4]


def test_composable_nested_groups():
    # Outer list, inner product
    config = SweepConfig(
        type="list",
        groups=[
            {"type": "product", "groups": [{"params": {"a": [1]}}, {"params": {"b": [2, 3]}}]},
            {"params": {"a": [4], "b": [5]}},
        ],
        base_values={},
    )
    points = expand_sweep(config)
    # Group 1: 1 * 2 = 2 points (a=1, b=2), (a=1, b=3)
    # Group 2: 1 * 1 = 1 point (a=4, b=5)
    # Total: 3 points
    assert len(points) == 3

    assert any(p.parameters["a"] == 1 and p.parameters["b"] == 2 for p in points)
    assert any(p.parameters["a"] == 1 and p.parameters["b"] == 3 for p in points)
    assert any(p.parameters["a"] == 4 and p.parameters["b"] == 5 for p in points)


def test_group_defaults():
    config = SweepConfig(
        type="product",
        groups=[{"defaults": {"d": 10}, "params": {"a": [1]}}, {"params": {"b": [2]}}],
        base_values={"base": 0},
    )
    points = expand_sweep(config)
    assert len(points) == 1
    p = points[0]
    assert p.parameters["d"] == 10
    assert p.parameters["a"] == 1
    assert p.parameters["b"] == 2
    assert p.parameters["base"] == 0


def test_composable_list_configs():
    config = SweepConfig(
        type="list", groups=[{"configs": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]}], base_values={}
    )
    points = expand_sweep(config)
    assert len(points) == 2
    assert points[0].parameters["a"] == 1
    assert points[1].parameters["a"] == 3


def test_list_composition_basic():
    """Test basic list composition in product mode."""
    config = SweepConfig(
        type="product",
        groups=[{"params": {"subconfig": ["a", "b"]}}, {"params": {"subconfig": ["c", "d"]}}],
        list_composition=["subconfig"],
    )
    points = expand_sweep(config)
    assert len(points) == 4  # 2 * 2

    # Each point should have subconfig as a list with 2 elements
    for point in points:
        assert isinstance(point.parameters["subconfig"], list)
        assert len(point.parameters["subconfig"]) == 2

    # Check specific combinations
    assert any(point.parameters["subconfig"] == ["a", "c"] for point in points)
    assert any(point.parameters["subconfig"] == ["a", "d"] for point in points)
    assert any(point.parameters["subconfig"] == ["b", "c"] for point in points)
    assert any(point.parameters["subconfig"] == ["b", "d"] for point in points)


def test_list_composition_with_normal_params():
    """Test list composition mixed with normal parameters."""
    config = SweepConfig(
        type="product",
        groups=[
            {"params": {"subconfig": ["a", "b"], "learning_rate": [0.001, 0.01]}},
            {"params": {"subconfig": ["c"], "batch_size": [32, 64]}},
        ],
        list_composition=["subconfig"],
    )
    points = expand_sweep(config)
    # (2 subconfig * 2 lr) * (1 subconfig * 2 batch_size) = 8
    assert len(points) == 8

    # Check that subconfig is accumulated as list
    for point in points:
        assert isinstance(point.parameters["subconfig"], list)
        assert len(point.parameters["subconfig"]) == 2
        assert point.parameters["subconfig"][1] == "c"
        assert point.parameters["subconfig"][0] in ["a", "b"]

    # Check that normal params still work (last value wins)
    for point in points:
        assert point.parameters["learning_rate"] in [0.001, 0.01]
        assert point.parameters["batch_size"] in [32, 64]


def test_list_composition_three_groups():
    """Test list composition across three groups."""
    config = SweepConfig(
        type="product",
        groups=[
            {"params": {"subconfig": ["a"]}},
            {"params": {"subconfig": ["b"]}},
            {"params": {"subconfig": ["c"]}},
        ],
        list_composition=["subconfig"],
    )
    points = expand_sweep(config)
    assert len(points) == 1

    # Should accumulate all three values
    assert points[0].parameters["subconfig"] == ["a", "b", "c"]


def test_list_composition_single_value():
    """Test list composition with single value per group."""
    config = SweepConfig(
        type="product",
        groups=[{"params": {"subconfig": ["a"]}}, {"params": {"other": ["x"]}}],
        list_composition=["subconfig"],
    )
    points = expand_sweep(config)
    assert len(points) == 1

    # subconfig should still be a list even with one value
    assert points[0].parameters["subconfig"] == ["a"]
    assert points[0].parameters["other"] == "x"


def test_list_composition_empty():
    """Test with empty list_composition."""
    config = SweepConfig(
        type="product",
        groups=[{"params": {"subconfig": ["a"]}}, {"params": {"subconfig": ["b"]}}],
        list_composition=[],
    )
    points = expand_sweep(config)
    assert len(points) == 1

    # Without list_composition, last value wins
    assert points[0].parameters["subconfig"] == "b"


def test_list_composition_in_list_mode():
    """Test that list composition in list mode only affects each group
    individually."""
    config = SweepConfig(
        type="list",
        groups=[{"params": {"subconfig": ["a"]}}, {"params": {"subconfig": ["b"]}}],
        list_composition=["subconfig"],
    )
    points = expand_sweep(config)
    # List mode concatenates, not cross-product
    assert len(points) == 2

    # In list mode, groups don't merge, so list composition doesn't accumulate across groups
    # Each point has only its own value (not as a list, since there's no merging)
    assert points[0].parameters["subconfig"] == "a"
    assert points[1].parameters["subconfig"] == "b"


def test_list_composition_multiple_params():
    """Test list composition with multiple parameters."""
    config = SweepConfig(
        type="product",
        groups=[
            {"params": {"subconfig": ["a"], "plugin": ["p1"]}},
            {"params": {"subconfig": ["b"], "plugin": ["p2"]}},
        ],
        list_composition=["subconfig", "plugin"],
    )
    points = expand_sweep(config)
    assert len(points) == 1

    # Both params should be accumulated
    assert points[0].parameters["subconfig"] == ["a", "b"]
    assert points[0].parameters["plugin"] == ["p1", "p2"]


def test_list_composition_nested_groups():
    """Test list composition with nested groups."""
    config = SweepConfig(
        type="product",
        groups=[
            {"params": {"subconfig": ["a"]}},
            {
                "type": "product",
                "groups": [{"params": {"subconfig": ["b"]}}, {"params": {"subconfig": ["c"]}}],
            },
        ],
        list_composition=["subconfig"],
    )
    points = expand_sweep(config)
    assert len(points) == 1

    # Should accumulate across nested groups
    # The nested product creates one combination with both "b" and "c"
    # Then combined with "a" from the outer group
    assert points[0].parameters["subconfig"] == ["a", "b", "c"]


def test_list_composition_with_defaults():
    """Test list composition with group defaults."""
    config = SweepConfig(
        type="product",
        groups=[
            {"defaults": {"subconfig": "default"}, "params": {"a": [1]}},
            {"params": {"subconfig": ["x"]}},
        ],
        list_composition=["subconfig"],
    )
    points = expand_sweep(config)
    assert len(points) == 1

    # Default value should be accumulated with the sweep value
    assert points[0].parameters["subconfig"] == ["default", "x"]


def test_list_composition_with_list_value():
    """Test that list values are properly flattened when accumulated."""
    config = SweepConfig(
        type="product",
        groups=[
            {"params": {"subconfig": [["a", "b"]]}},  # List value - gets flattened
            {"params": {"subconfig": ["c"]}},
        ],
        list_composition=["subconfig"],
    )
    points = expand_sweep(config)
    assert len(points) == 1

    # List values are flattened when accumulated (fixed behavior)
    assert points[0].parameters["subconfig"] == ["a", "b", "c"]
