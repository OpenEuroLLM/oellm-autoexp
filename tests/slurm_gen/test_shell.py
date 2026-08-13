"""Tests for shell utilities."""

import errno
import subprocess

import pytest

import oellm_autoexp.slurm_gen.shell as shell_mod
from oellm_autoexp.slurm_gen.shell import run_command


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


class TestSpawnRetry:
    """Retry-on-transient-fork-failure behaviour (busy login node, EAGAIN)."""

    def _patch(self, monkeypatch, side_effects):
        """Patch subprocess.run to yield ``side_effects`` in order and record
        call count; also make sleep instant."""
        calls = {"n": 0}

        def fake_run(*args, **kwargs):
            i = calls["n"]
            calls["n"] += 1
            effect = side_effects[min(i, len(side_effects) - 1)]
            if isinstance(effect, Exception):
                raise effect
            return effect

        monkeypatch.setattr(shell_mod.subprocess, "run", fake_run)
        monkeypatch.setattr(shell_mod.time, "sleep", lambda _s: None)
        return calls

    def test_retries_transient_eagain_then_succeeds(self, monkeypatch):
        ok = subprocess.CompletedProcess(["sbatch"], 0, stdout="Submitted 123", stderr="")
        calls = self._patch(
            monkeypatch,
            [BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable")] * 2 + [ok],
        )
        result = run_command(["sbatch", "job.sh"], spawn_retry_backoff=0.0)
        assert result.stdout == "Submitted 123"
        assert calls["n"] == 3  # 2 failures + 1 success

    def test_gives_up_after_max_retries(self, monkeypatch):
        calls = self._patch(
            monkeypatch, [BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable")]
        )
        with pytest.raises(BlockingIOError):
            run_command(["sbatch", "job.sh"], max_spawn_retries=3, spawn_retry_backoff=0.0)
        assert calls["n"] == 4  # initial attempt + 3 retries

    def test_permanent_oserror_not_retried(self, monkeypatch):
        # ENOENT (command not found) must propagate immediately, no retries.
        calls = self._patch(monkeypatch, [FileNotFoundError(errno.ENOENT, "No such file")])
        with pytest.raises(FileNotFoundError):
            run_command(["nonexistent_cmd"], spawn_retry_backoff=0.0)
        assert calls["n"] == 1
