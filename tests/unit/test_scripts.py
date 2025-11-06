import json
from pathlib import Path

import argparse
import pytest

from types import SimpleNamespace

from scripts import run_autoexp_container
from scripts.monitor_autoexp import (
    _apply_monitor_overrides,
    _load_session_target,
    _resolve_session_path,
)


def test_run_autoexp_container_submits_with_fake_slurm(monkeypatch, tmp_path: Path):
    """Ensure the container wrapper executes plan inside the image and submit
    on host with fake SLURM."""

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return argparse.Namespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_autoexp_container, "run_with_tee", fake_run)

    args = argparse.Namespace(
        image="image.sif",
        apptainer_cmd="apptainer",
        config_ref="autoexp",
        config_dir=str(tmp_path),
        override=["project=demo"],
        dry_run=False,
        monitor_only=False,
        no_monitor=True,
        no_submit=False,
        debug=False,
        verbose=False,
        env=[],
        bind=[],
        ihelp=False,
        use_fake_slurm=True,
        manifest=tmp_path / "plan.json",
        plan_id=None,
    )
    monkeypatch.setattr(run_autoexp_container, "parse_args", lambda _=None: args)

    with pytest.raises(SystemExit) as exc:
        run_autoexp_container.main()

    assert exc.value.code == 0

    # Expect two calls: container plan + host submit
    assert len(calls) == 2
    assert calls[0][0] == args.apptainer_cmd
    assert "plan_autoexp.py" in calls[0][-1]
    assert calls[1][0] == "python"
    assert any(part.endswith("scripts/submit_autoexp.py") for part in calls[1])
    assert "--use-fake-slurm" in calls[1]
    assert "--no-monitor" in calls[1]


def test_run_autoexp_container_infers_manifest_path(monkeypatch, tmp_path: Path):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        if "plan_autoexp.py" in " ".join(cmd):
            return argparse.Namespace(
                returncode=0,
                stdout=f"Plan manifest written to: {tmp_path}/outputs/manifests/plan_20240101-000000_abcdef.json\n",
                stderr="",
            )
        calls.append(cmd)
        return argparse.Namespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv("SLURM_ACCOUNT", "demo")
    monkeypatch.setattr(run_autoexp_container, "run_with_tee", fake_run)

    args = argparse.Namespace(
        image=None,
        apptainer_cmd="apptainer",
        config_ref="autoexp",
        config_dir=str(tmp_path),
        override=[],
        dry_run=True,
        monitor_only=False,
        no_monitor=True,
        no_submit=False,
        debug=False,
        verbose=False,
        env=[],
        bind=[],
        ihelp=False,
        use_fake_slurm=False,
        manifest=None,
        plan_id=None,
    )
    monkeypatch.setattr(run_autoexp_container, "parse_args", lambda _=None: args)

    with pytest.raises(SystemExit) as exc:
        run_autoexp_container.main()

    assert exc.value.code == 0
    assert calls
    submit_cmd = calls[0]
    assert submit_cmd[0] == "python"
    assert any(part.endswith("scripts/submit_autoexp.py") for part in submit_cmd)
    assert any(str(tmp_path) in part for part in submit_cmd)


def test_run_autoexp_container_monitor_only(monkeypatch, tmp_path: Path):
    """Monitor-only mode should directly invoke the host monitor helper."""

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return argparse.Namespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_autoexp_container, "run_with_tee", fake_run)

    args = argparse.Namespace(
        image=None,
        apptainer_cmd="apptainer",
        config_ref="autoexp",
        config_dir=str(tmp_path),
        override=[],
        dry_run=False,
        monitor_only=True,
        no_monitor=False,
        no_submit=False,
        debug=False,
        verbose=True,
        env=[],
        bind=[],
        ihelp=False,
        use_fake_slurm=False,
        manifest=tmp_path / "plan.json",
        plan_id=None,
    )
    monkeypatch.setattr(run_autoexp_container, "parse_args", lambda _=None: args)

    with pytest.raises(SystemExit) as exc:
        run_autoexp_container.main()

    assert exc.value.code == 0

    assert len(calls) == 1
    assert calls[0][0] == "python"
    assert calls[0][1].endswith("scripts/monitor_autoexp.py")
    assert "--manifest" in calls[0]
    assert "--verbose" in calls[0]


def test_build_container_command_includes_env_and_bind():
    args = argparse.Namespace(
        apptainer_cmd="singularity",
        image="container.sif",
        env=["FOO=1", "BAR=2"],
        bind=["/data:/data"],
    )
    cmd = run_autoexp_container._build_container_command(args, ["python", "script.py"])
    assert "--bind" in cmd
    assert "--env" in cmd
    assert cmd[-1].endswith("script.py")


def test_apply_monitor_overrides_sets_direct_key():
    config = {
        "class_name": "NullMonitor",
        "poll_interval_seconds": 600,
        "inactivity_threshold_seconds": 900,
        "termination_string": None,
        "termination_command": None,
        "log_path_template": "{output_dir}/slurm.out",
        "output_paths": [],
        "start_condition_cmd": None,
        "start_condition_interval_seconds": None,
        "debug_sync": False,
    }
    spec = SimpleNamespace(config=config)
    _apply_monitor_overrides(spec, ["debug_sync=true"])
    assert spec.config["debug_sync"] in (True, "true")


def test_resolve_session_path_accepts_ids(tmp_path: Path) -> None:
    state_dir = tmp_path / "monitor"
    state_dir.mkdir()
    session_file = state_dir / "abc123.json"
    session_file.write_text(json.dumps({"session_id": "abc123", "manifest_path": "plan.json"}))

    resolved = _resolve_session_path("abc123", state_dir)
    assert resolved == session_file.resolve()


def test_load_session_target_reads_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "plan.json"
    manifest.write_text("{}", encoding="utf-8")
    session_file = tmp_path / "session.json"
    session_file.write_text(json.dumps({"session_id": "sess", "manifest_path": str(manifest)}))

    target = _load_session_target(session_file)
    assert target.session_id == "sess"
    assert target.manifest_path == manifest.resolve()
    assert target.session_path == session_file.resolve()
