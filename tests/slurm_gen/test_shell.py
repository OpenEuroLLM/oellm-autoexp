"""Tests for shell utilities."""

import subprocess

import pytest

from slurm_gen.shell import run_command


class TestRunCommand:
    """Tests for run_command function."""

    def test_successful_command(self):
        """Test running a successful command."""
        result = run_command(["echo", "hello"])
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_command_with_args(self):
        """Test command with multiple arguments."""
        result = run_command(["echo", "-n", "test"])
        assert result.returncode == 0
        assert "test" in result.stdout

    def test_failing_command(self):
        """Test a command that returns non-zero."""
        result = run_command(["false"])
        assert result.returncode != 0

    def test_check_raises(self):
        """Test that check=True raises on failure."""
        with pytest.raises(subprocess.CalledProcessError):
            run_command(["false"], check=True)

    def test_captures_stderr(self):
        """Test that stderr is captured."""
        result = run_command(["ls", "/nonexistent_path_12345"])
        assert result.returncode != 0
        assert result.stderr  # Should have error message

    def test_captures_stdout(self):
        """Test that stdout is captured."""
        result = run_command(["echo", "output"])
        assert "output" in result.stdout

    def test_timeout(self):
        """Test command timeout."""
        with pytest.raises(subprocess.TimeoutExpired):
            run_command(["sleep", "10"], timeout=0.1)
