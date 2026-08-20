from __future__ import annotations

import pytest

from oellm_autoexp.backends.oellm_eval_backend import (
    OELLMEvalBackend,
    OELLMEvalBackendConfig,
)


def test_builds_command_with_task_groups():
    cfg = OELLMEvalBackendConfig(
        models=["/path/to/hf"],
        task_groups=["open-sci-0.01"],
        venv_path="/home/u/venv",
    )
    cmd = OELLMEvalBackend(cfg).build_launch_command()
    assert cmd.startswith("oellm-eval schedule")
    assert "--models /path/to/hf" in cmd
    assert "--task_groups open-sci-0.01" in cmd
    assert "--venv_path /home/u/venv" in cmd
    assert "--local true" in cmd


def test_builds_command_with_tasks_and_nshot():
    cfg = OELLMEvalBackendConfig(
        models=["m1", "m2"],
        tasks=["hellaswag", "mmlu"],
        n_shot=5,
        venv_path="/v",
    )
    cmd = OELLMEvalBackend(cfg).build_launch_command()
    assert "--models m1,m2" in cmd
    assert "--tasks hellaswag,mmlu" in cmd
    assert "--n_shot 5" in cmd


def test_builds_command_with_eval_csv():
    cfg = OELLMEvalBackendConfig(
        eval_csv_path="/some/path.csv",
        venv_path="/v",
    )
    cmd = OELLMEvalBackend(cfg).build_launch_command()
    assert "--eval_csv_path /some/path.csv" in cmd
    assert "--models" not in cmd
    assert "--task_groups" not in cmd


def test_validate_requires_models_or_csv():
    OELLMEvalBackend(
        OELLMEvalBackendConfig(venv_path="/v", task_groups=["foo"], models=["m"])
    ).validate()  # ok
    with pytest.raises(ValueError, match="`models` is required"):
        OELLMEvalBackend(
            OELLMEvalBackendConfig(venv_path="/v", task_groups=["foo"], models=[])
        ).validate()


def test_validate_eval_csv_excludes_other_args():
    cfg = OELLMEvalBackendConfig(
        eval_csv_path="/p.csv",
        models=["m"],
        venv_path="/v",
    )
    with pytest.raises(ValueError, match="cannot combine"):
        OELLMEvalBackend(cfg).validate()


def test_validate_tasks_requires_nshot():
    cfg = OELLMEvalBackendConfig(
        models=["m"],
        tasks=["t"],
        venv_path="/v",
    )
    with pytest.raises(ValueError, match="`n_shot` is required"):
        OELLMEvalBackend(cfg).validate()


def test_validate_local_requires_venv():
    cfg = OELLMEvalBackendConfig(
        models=["m"],
        task_groups=["g"],
        local=True,
        venv_path=None,
    )
    with pytest.raises(ValueError, match="requires `venv_path`"):
        OELLMEvalBackend(cfg).validate()


def test_full_cmd_is_overridable():
    cfg = OELLMEvalBackendConfig(
        models=["m"],
        task_groups=["g"],
        venv_path="/v",
        full_cmd="echo custom-command",
    )
    assert OELLMEvalBackend(cfg).build_launch_command() == "echo custom-command"


def test_extra_cli_args_passthrough():
    cfg = OELLMEvalBackendConfig(
        models=["m"],
        task_groups=["g"],
        venv_path="/v",
        extra_cli_args=["--foo=bar", "--baz"],
    )
    cmd = OELLMEvalBackend(cfg).build_launch_command()
    assert "--foo=bar" in cmd
    assert "--baz" in cmd
