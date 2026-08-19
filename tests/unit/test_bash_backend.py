"""Tests for the BashBackend and the auto_cancel test sweep.

BashBackend runs an arbitrary bash string in place of a real trainer so the
job-control configs (config/job/auto_cancel.yaml) can be exercised on a real
cluster -- emulating a clean finish, a restart-on-error, a time-limit, a silent
stall (cancel), and the srun-error spam the monitor fails to cancel -- without
launching megatron. See config/backend/bash.yaml and
config/experiments/tests/auto_cancel_sweep.yaml.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oellm_autoexp.backends.base import BashBackend, BashBackendConfig

REPO_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def test_bash_backend_returns_command_verbatim():
    cfg = BashBackendConfig(command='echo "hi"\nsleep 5', env={"FOO": "1"})
    backend = BashBackend(cfg)
    backend.validate()
    assert backend.build_launch_command() == 'echo "hi"\nsleep 5'


def test_bash_backend_rejects_empty_command():
    backend = BashBackend(BashBackendConfig(command="   "))
    with pytest.raises(ValueError, match="non-empty"):
        backend.validate()


def test_bash_backend_rejects_single_quotes():
    # The sbatch template wraps the command in single quotes, so a single quote
    # in the command would break out of `bash -c '...'`.
    backend = BashBackend(BashBackendConfig(command="echo 'oops'"))
    with pytest.raises(ValueError, match="single quote"):
        backend.validate()


def test_auto_cancel_sweep_expands_to_per_event_bash_jobs():
    """The test sweep expands to one BashBackend job per auto_cancel event,
    each carrying the bash command (multi-line, swept through Hydra) that
    triggers it."""
    from oellm_autoexp.config.loader import load_config_reference
    from oellm_autoexp.config.schema import BackendInterface, ConfigSetup, RootConfig
    from oellm_autoexp.orchestrator import build_execution_plan

    setup = ConfigSetup(
        pwd=".",
        config_name="experiments/tests/auto_cancel_sweep",
        config_dir=str(REPO_CONFIG_DIR),
    )
    root = load_config_reference(config_setup=setup, config_class=RootConfig)
    plan = build_execution_plan(root, setup)

    cmds = {}
    for job in plan.jobs:
        assert job.config.backend.class_name == "BashBackend"
        assert job.config.slurm.launcher_cmd == ""  # no container wrapper
        backend = job.config.backend.instantiate(BackendInterface)
        backend.validate()
        cmds[job.config.job.name] = backend.build_launch_command()

    # One job per scenario, each with the marker that triggers its event.
    assert "[after training is done]" in cmds["auto_cancel__finish_training"]
    assert "on test set" in cmds["auto_cancel__finish_eval"]
    assert "Exited with exit code 1" in cmds["auto_cancel__restart_on_error"]
    assert "DUE TO TIME LIMIT" in cmds["auto_cancel__finish_on_time_limit"]
    assert "sleep " in cmds["auto_cancel__cancel_on_silent_stall"]
    assert "srun: error" in cmds["auto_cancel__srun_spam_not_cancelled"]

    # The events the sweep targets are actually present in auto_cancel.yaml.
    events = {e.name: e.action.class_name for e in plan.jobs[0].config.job.log_events}
    assert events["inactive"] == "CancelAction"
    assert "error" in events

    # The silent-stall job overrides the inactive thresholds per-job (so a fresh
    # plan bakes them into its state file) while others keep the defaults.
    by_name = {j.config.job.name: j for j in plan.jobs}

    def _inactive(job):
        return next(e for e in job.config.job.log_events if e.name == "inactive")

    # The override is POSITIONAL (`job.log_events.<i>.inactivity_*`), because
    # Hydra cannot address a list element by name. That coupling has broken
    # once already: index 4 was `inactive` when the sweep was written and became
    # `err_fp8_metadata` after the error patterns were added, so the override
    # silently landed on the wrong event. Recompute it here so a future drift
    # fails with the index to use rather than with a bare value mismatch.
    _assert_stall_override_targets_inactive()

    stall = _inactive(by_name["auto_cancel__cancel_on_silent_stall"])
    assert (stall.inactivity_polls, stall.inactivity_timeout_s) == (1, 60.0)

    # The other jobs keep whatever auto_cancel.yaml currently sets; read it from
    # the file rather than pinning a literal, so tuning the production policy
    # does not fail an unrelated test.
    default = _inactive(by_name["auto_cancel__finish_training"])
    policy_inactive = _auto_cancel_events()[1]
    assert (default.inactivity_polls, default.inactivity_timeout_s) == (
        policy_inactive["inactivity_polls"],
        float(policy_inactive["inactivity_timeout_s"]),
    )


def _auto_cancel_events() -> tuple[int, dict]:
    """Return (index, event) of the `inactive` entry in auto_cancel.yaml."""
    import yaml

    policy = yaml.safe_load((REPO_CONFIG_DIR / "job" / "auto_cancel.yaml").read_text())
    for idx, event in enumerate(policy["log_events"]):
        if event["name"] == "inactive":
            return idx, event
    raise AssertionError("auto_cancel.yaml has no 'inactive' log event")


def _assert_stall_override_targets_inactive() -> None:
    """The sweep's positional override must point at the `inactive` event."""
    import re

    import yaml

    expected_idx, _ = _auto_cancel_events()
    sweep_text = (REPO_CONFIG_DIR / "experiments" / "tests" / "auto_cancel_sweep.yaml").read_text()
    used = {int(m) for m in re.findall(r"job\.log_events\.(\d+)\.inactivity_", sweep_text)}
    assert used, "auto_cancel_sweep.yaml no longer overrides the inactivity thresholds"
    assert used == {expected_idx}, (
        f"auto_cancel_sweep.yaml pokes log_events index {sorted(used)} but 'inactive' is now "
        f"at index {expected_idx} in config/job/auto_cancel.yaml — the override is landing on "
        f"'{yaml.safe_load((REPO_CONFIG_DIR / 'job' / 'auto_cancel.yaml').read_text())['log_events'][min(used)]['name']}'"
    )
