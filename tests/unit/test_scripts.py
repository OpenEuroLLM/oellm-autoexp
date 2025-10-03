from pathlib import Path

import argparse

from scripts import run_megatron_container


def test_run_megatron_container_invokes_fake_submit(monkeypatch, tmp_path: Path):
    calls = []

    def fake_run(args, cmd_parts):
        calls.append(cmd_parts)
        return 0

    monkeypatch.setattr(run_megatron_container, "_run_in_container", fake_run)

    args = argparse.Namespace(
        image="image.sif",
        apptainer_cmd="apptainer",
        config_ref="autoexp",
        config_dir=str(tmp_path),
        override=["project=demo"],
        dry_run=False,
        fake_submit=True,
        show_command=False,
    )
    monkeypatch.setattr(run_megatron_container, "parse_args", lambda: args)

    run_megatron_container.main()

    assert len(calls) == 2
    assert calls[0][-1] == "--dry-run"
    assert calls[1][-1] == "--use-fake-slurm"
