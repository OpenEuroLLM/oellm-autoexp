"""Tests for command-line formatting functions in dag_resolver."""

from oellm_autoexp.hydra_staged_sweep.dag_resolver import param_to_cmdlines, config_to_cmdline


def test_param_to_cmdlines_string():
    """Test formatting a simple string parameter."""
    result = param_to_cmdlines("key", "value", prefix="++")
    assert result == ['++key="value"']


def test_param_to_cmdlines_string_with_quotes():
    """Test formatting a string with quotes."""
    result = param_to_cmdlines("key", 'value"with"quotes', prefix="++")
    assert result == ['++key="value\\"with\\"quotes"']


def test_param_to_cmdlines_list_of_strings():
    """Test formatting a list of strings (config group list)."""
    result = param_to_cmdlines("subconfig", ["a", "b", "c"], prefix="++")
    assert result == ["++subconfig=[a,b,c]"]


def test_param_to_cmdlines_list_of_strings_empty():
    """Test formatting an empty list of strings."""
    result = param_to_cmdlines("subconfig", [], prefix="++")
    assert result == ["++subconfig=[]"]


def test_param_to_cmdlines_list_of_strings_single():
    """Test formatting a single-element list of strings."""
    result = param_to_cmdlines("subconfig", ["a"], prefix="++")
    assert result == ["++subconfig=[a]"]


def test_param_to_cmdlines_dict():
    """Test formatting a dict parameter (falls through to
    config_to_cmdline)."""
    result = param_to_cmdlines("nested", {"a": 1, "b": 2}, prefix="++")
    # Should expand into nested key=value pairs
    assert "++nested.a=1" in result
    assert "++nested.b=2" in result


def test_param_to_cmdlines_list_of_dicts():
    """Test formatting a list of dicts (not a config group list)."""
    result = param_to_cmdlines("items", [{"a": 1}, {"b": 2}], prefix="++")
    # Should use indexed format with list creation line
    assert "++items=[0,1]" in result
    assert "++items.0.a=1" in result
    assert "++items.1.b=2" in result


def test_param_to_cmdlines_list_mixed_types():
    """Test formatting a list with mixed types (not all strings)."""
    result = param_to_cmdlines("mixed", ["a", 1, "b"], prefix="++")
    # Should fall through to config_to_cmdline with list creation line
    assert "++mixed=[0,1,2]" in result


def test_config_to_cmdline_simple():
    """Test config_to_cmdline with simple values."""
    result = config_to_cmdline({"a": 1, "b": "text"}, override="++")
    assert "++a=1" in result
    assert '++b="text"' in result


def test_config_to_cmdline_nested():
    """Test config_to_cmdline with nested dict."""
    result = config_to_cmdline({"outer": {"inner": "value"}}, override="++")
    assert '++outer.inner="value"' in result


def test_config_to_cmdline_list():
    """Test config_to_cmdline with list."""
    result = config_to_cmdline({"items": ["a", "b"]}, override="++")
    assert "++items=[0,1]" in result
    assert '++items.0="a"' in result
    assert '++items.1="b"' in result


def test_config_to_cmdline_null():
    """Test config_to_cmdline with null value."""
    result = config_to_cmdline({"key": None}, override="++")
    assert "++key=null" in result
