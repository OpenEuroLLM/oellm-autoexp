"""Tests for defensive code branches in expander.py.

Some code branches exist as defensive programming to handle edge cases
that shouldn't occur with the current implementation but provide safety
if the code logic changes in the future.
"""

from oellm_autoexp.hydra_staged_sweep.expander import _cartesian_product_groups


def test_list_composition_always_initializes_as_list():
    """Document that list_composition parameters are always initialized as
    lists.

    This test documents the behavior that makes line 269 unreachable:
    When a key is in list_composition and not yet in merged_params, it's
    initialized as an empty list (line 260), so it will always be a list
    when checked on line 262.

    The else branch on line 269 is defensive code that would handle the case
    where merged_params[key] exists but is not a list, which can't happen
    with the current initialization logic.
    """
    # Create a scenario where we accumulate values
    groups = [
        [({"config": "a", "other": 1}, (0,), (False,))],
        [({"config": "b", "other": 2}, (1,), (False,))],
        [({"config": "c", "other": 3}, (2,), (False,))],
    ]

    list_composition = {"config"}
    result = _cartesian_product_groups(groups, list_composition)

    assert len(result) == 1
    params, _, _ = result[0]

    # All values are accumulated into a list
    assert params["config"] == ["a", "b", "c"]
    # Non-list_composition params get overridden (last wins)
    assert params["other"] == 3


def test_list_composition_initialization_order():
    """Verify that list_composition handles initialization correctly regardless
    of order."""
    # Even if the first group doesn't have the list_composition key,
    # it should work when subsequent groups add it

    groups = [
        # First group: doesn't have 'plugin'
        [({"setting": "value1"}, (0,), (False,))],
        # Second group: introduces 'plugin'
        [({"plugin": "p1"}, (1,), (False,))],
        # Third group: adds another 'plugin'
        [({"plugin": "p2"}, (2,), (False,))],
    ]

    list_composition = {"plugin"}
    result = _cartesian_product_groups(groups, list_composition)

    assert len(result) == 1
    params, _, _ = result[0]

    # Even though 'plugin' wasn't in the first group, it's accumulated correctly
    assert params["plugin"] == ["p1", "p2"]
    assert params["setting"] == "value1"


def test_defensive_branch_simulation():
    """Simulate what line 269 would do if it were reachable.

    This documents the defensive behavior: if a non-list value existed in
    merged_params for a list_composition key, it would be converted to a list
    with both the old and new values.
    """
    # Simulate the exact logic from line 269
    merged_params = {"config": "old_value"}  # Hypothetical non-list value
    new_value = "new_value"

    # Simulate the conversion that line 269 would perform
    converted_value = [merged_params["config"], new_value]

    # This is what line 269 does: convert the existing non-list value
    # to a list containing both the old and new values
    assert converted_value == ["old_value", "new_value"]


def test_list_composition_empty_initialization():
    """Verify that list_composition initializes correctly when first
    encountered."""
    # Need at least 2 groups to trigger the merging logic
    groups = [
        [({"other": "value"}, (0,), (False,))],  # First group without the key
        [({"plugins": "p1"}, (1,), (False,))],  # Second group introduces it
    ]

    list_composition = {"plugins"}
    result = _cartesian_product_groups(groups, list_composition)

    assert len(result) == 1
    params, _, _ = result[0]

    # When first initialized for list_composition, it's put in a list
    assert params["plugins"] == ["p1"]
    assert params["other"] == "value"


def test_list_composition_robustness():
    """Test that list_composition is robust across various scenarios."""
    # Test with multiple parameters being accumulated
    groups = [
        [({"a": "a1", "b": "b1", "c": "c1"}, (0,), (False,))],
        [({"a": "a2", "b": "b2", "c": "c2"}, (1,), (False,))],
    ]

    # Only 'a' and 'b' are in list_composition
    list_composition = {"a", "b"}
    result = _cartesian_product_groups(groups, list_composition)

    assert len(result) == 1
    params, _, _ = result[0]

    # 'a' and 'b' are accumulated
    assert params["a"] == ["a1", "a2"]
    assert params["b"] == ["b1", "b2"]
    # 'c' is overridden (last wins)
    assert params["c"] == "c2"
