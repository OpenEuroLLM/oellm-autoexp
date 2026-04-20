"""Tests for config group detection and smart prefix selection."""

from pathlib import Path
import tempfile


from hydra_staged_sweep.dag_resolver import is_config_group, param_to_cmdlines


def test_is_config_group_with_slash():
    """Test that keys with '/' are detected as config groups."""
    assert is_config_group("basic/subconfig", None)
    assert is_config_group("db/mysql", None)
    assert is_config_group("nested/path/config", None)


def test_is_config_group_directory_exists():
    """Test that existing directories are detected as config groups."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)

        # Create a config group directory
        (config_dir / "database").mkdir()

        assert is_config_group("database", config_dir)
        assert not is_config_group("nonexistent", config_dir)


def test_is_config_group_no_config_dir():
    """Test that without config_dir, only slash detection works."""
    assert is_config_group("basic/subconfig", None)
    assert not is_config_group("regular_param", None)


def test_is_config_group_file_not_directory():
    """Test that files are not detected as config groups."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)

        # Create a file, not a directory
        (config_dir / "config.yaml").touch()

        assert not is_config_group("config.yaml", config_dir)


def test_param_to_cmdlines_string():
    """Test string parameter conversion."""
    result = param_to_cmdlines("key", "value", prefix="++")
    assert result == ['++key="value"']


def test_param_to_cmdlines_string_with_quotes():
    """Test string with quotes is properly escaped."""
    result = param_to_cmdlines("key", 'value with "quotes"', prefix="++")
    assert result == ['++key="value with \\"quotes\\""']


def test_param_to_cmdlines_string_list_format():
    """Test string that looks like a list is passed through."""
    result = param_to_cmdlines("key", "[a,b,c]", prefix="++")
    assert result == ["++key=[a,b,c]"]


def test_param_to_cmdlines_string_list():
    """Test list of strings is formatted correctly."""
    result = param_to_cmdlines("subconfig", ["a", "b", "c"], prefix="")
    assert result == ["subconfig=[a,b,c]"]


def test_param_to_cmdlines_string_list_with_prefix():
    """Test list of strings with prefix."""
    result = param_to_cmdlines("plugins", ["p1", "p2"], prefix="++")
    assert result == ["++plugins=[p1,p2]"]


def test_param_to_cmdlines_single_element_list():
    """Test list with single element."""
    result = param_to_cmdlines("subconfig", ["single"], prefix="")
    assert result == ["subconfig=[single]"]


def test_param_to_cmdlines_empty_string():
    """Test empty string parameter."""
    result = param_to_cmdlines("key", "", prefix="++")
    assert result == ['++key=""']


def test_param_to_cmdlines_with_config_dir():
    """Test that config_dir parameter is accepted (even if not used in current
    logic)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = param_to_cmdlines("key", "value", prefix="++", config_dir=tmpdir)
        assert result == ['++key="value"']


def test_param_to_cmdlines_list_empty():
    """Test empty list."""
    result = param_to_cmdlines("key", [], prefix="++")
    assert result == ["++key=[]"]


def test_param_to_cmdlines_preserves_prefix():
    """Test that prefix is preserved correctly."""
    # With prefix
    result1 = param_to_cmdlines("key", ["a", "b"], prefix="++")
    assert result1[0].startswith("++")

    # Without prefix
    result2 = param_to_cmdlines("key", ["a", "b"], prefix="")
    assert not result2[0].startswith("++")


def test_config_group_detection_real_directory_structure():
    """Test with realistic Hydra config directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)

        # Create typical Hydra config groups
        (config_dir / "db").mkdir()
        (config_dir / "server").mkdir()
        (config_dir / "model").mkdir()

        # Create config files
        (config_dir / "db" / "mysql.yaml").touch()
        (config_dir / "db" / "postgres.yaml").touch()
        (config_dir / "config.yaml").touch()

        # Test detection
        assert is_config_group("db", config_dir)
        assert is_config_group("server", config_dir)
        assert is_config_group("model", config_dir)

        # Config file is not a group
        assert not is_config_group("config.yaml", config_dir)

        # Non-existent is not a group
        assert not is_config_group("optimizer", config_dir)

        # Nested paths are always groups (due to '/')
        assert is_config_group("db/mysql", config_dir)


def test_param_to_cmdlines_list_with_mixed_types():
    """Test that only all-string lists are formatted as [a,b,c]."""
    # All strings - should format as list
    result1 = param_to_cmdlines("key", ["a", "b"], prefix="")
    assert result1 == ["key=[a,b]"]

    # Note: Mixed types would fall through to config_to_cmdline,
    # but we're testing the string list path specifically


def test_is_config_group_path_object():
    """Test that Path objects work as config_dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        (config_dir / "testgroup").mkdir()

        # Test with Path object
        assert is_config_group("testgroup", config_dir)

        # Test with string path
        assert is_config_group("testgroup", str(config_dir))


def test_param_to_cmdlines_dict_value():
    """Test that dict values are passed to config_to_cmdline."""
    # Dict values should be handled by config_to_cmdline

    result = param_to_cmdlines("nested", {"a": 1, "b": 2}, prefix="++")

    # Should produce nested key overrides
    assert len(result) > 0
    assert all("nested." in r for r in result)


def test_param_to_cmdlines_number_value():
    """Test that number values are passed to config_to_cmdline."""
    result = param_to_cmdlines("count", 42, prefix="++")

    # Should produce a simple key=value override
    assert len(result) > 0


def test_param_to_cmdlines_bool_value():
    """Test that boolean values are passed to config_to_cmdline."""
    result = param_to_cmdlines("enabled", True, prefix="++")

    assert len(result) > 0


def test_param_to_cmdlines_list_with_non_strings():
    """Test that lists with non-string items are passed to
    config_to_cmdline."""
    result = param_to_cmdlines("numbers", [1, 2, 3], prefix="++")

    # Should handle as complex value, not as string list
    assert len(result) > 0
