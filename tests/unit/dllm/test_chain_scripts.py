"""Tests for the job shell scripts train_then_eval.sh and eval_chain.sh, with
stub commands.

The scripts run on the cluster around the GPU programs; here a stub stands in for each program, so
the tests check only the control flow: when the evaluation runs, which settings it gets from the
training flags, and which GPU each evaluation task gets.

Run with:
    pytest tests/unit/dllm/test_chain_scripts.py -v
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_THEN_EVAL = REPO_ROOT / "oellm_autoexp/dllm/train_then_eval.sh"
EVAL_CHAIN = REPO_ROOT / "oellm_autoexp/dllm/eval/eval_chain.sh"
_EVAL_KEYS = ("EVAL_CKPT", "EVAL_OBJECTIVE", "EVAL_BLOCK_SIZE", "EVAL_TOKEN_SHIFT", "EVAL_MASK_ID")


def _run(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    base = {key: value for key, value in os.environ.items() if not key.startswith("EVAL_")}
    base.pop("SLURM_NODEID", None)
    return subprocess.run(
        cmd, env={**base, **env}, cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )


def _train_flags(save: Path, *extra: str) -> list[str]:
    return [
        "true",  # stands in for torchrun and the launcher
        "--save", str(save), "--train-iters", "100", "--mask-token", "151669",
        "--bd-objective", "bd3", "--bd-block-size", "8", "--bd-block-schedule", "none",
        "--bd-token-shift", "0", *extra,
    ]  # fmt: skip


def _stub_eval(tmp_path: Path) -> tuple[str, Path]:
    """A stub eval chain that writes the EVAL_* settings it receives to a
    file."""
    out = tmp_path / "eval_env.txt"
    script = tmp_path / "stub_eval.sh"
    keys = " ".join(f'"{key}=${key}"' for key in (*_EVAL_KEYS, "EVAL_MODES"))
    script.write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' {keys} > {out}\n")
    return f"bash {script}", out


def _settings(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines())


def test_no_evaluation_before_the_last_iteration(tmp_path: Path) -> None:
    save = tmp_path / "checkpoints"
    save.mkdir()
    (save / "latest_checkpointed_iteration.txt").write_text("40")
    stub, out = _stub_eval(tmp_path)
    result = _run(["bash", str(TRAIN_THEN_EVAL), *_train_flags(save)], {"EVAL_CHAIN_CMD": stub})
    assert result.returncode == 0, result.stderr
    assert "no evaluation" in result.stdout
    assert not out.exists()


def test_evaluation_takes_its_settings_from_the_training_flags(tmp_path: Path) -> None:
    save = tmp_path / "checkpoints"
    save.mkdir()
    (save / "latest_checkpointed_iteration.txt").write_text("100")
    stub, out = _stub_eval(tmp_path)
    flags = _train_flags(save)
    flags[flags.index("--bd-block-schedule") + 1] = "0:4,0.15:16"
    flags[flags.index("--bd-token-shift") + 1] = "1"
    result = _run(["bash", str(TRAIN_THEN_EVAL), *flags], {"EVAL_CHAIN_CMD": stub})
    assert result.returncode == 0, result.stderr
    assert _settings(out) == {
        "EVAL_CKPT": str(save),
        "EVAL_OBJECTIVE": "bd3",
        "EVAL_BLOCK_SIZE": "16",  # last schedule entry, the block size at the end of training
        "EVAL_TOKEN_SHIFT": "1",
        "EVAL_MASK_ID": "151669",
        "EVAL_MODES": "",
    }


def test_ar_run_is_evaluated_with_block_size_one_and_ar_decoding(tmp_path: Path) -> None:
    save = tmp_path / "checkpoints"
    save.mkdir()
    (save / "latest_checkpointed_iteration.txt").write_text("100")
    stub, out = _stub_eval(tmp_path)
    flags = _train_flags(save)
    flags[flags.index("--bd-objective") + 1] = "ar"
    env = {"EVAL_CHAIN_CMD": stub, "EVAL_MODES": "static,dynamic"}
    result = _run(["bash", str(TRAIN_THEN_EVAL), *flags], env)
    assert result.returncode == 0, result.stderr
    settings = _settings(out)
    assert (settings["EVAL_OBJECTIVE"], settings["EVAL_BLOCK_SIZE"]) == ("ar", "1")
    assert (settings["EVAL_TOKEN_SHIFT"], settings["EVAL_MODES"]) == ("1", "ar")


def test_only_node_zero_evaluates(tmp_path: Path) -> None:
    save = tmp_path / "checkpoints"
    save.mkdir()
    (save / "latest_checkpointed_iteration.txt").write_text("100")
    stub, out = _stub_eval(tmp_path)
    env = {"EVAL_CHAIN_CMD": stub, "SLURM_NODEID": "1"}
    result = _run(["bash", str(TRAIN_THEN_EVAL), *_train_flags(save)], env)
    assert result.returncode == 0, result.stderr
    assert not out.exists()


def test_failed_training_stops_the_job(tmp_path: Path) -> None:
    stub, out = _stub_eval(tmp_path)
    flags = _train_flags(tmp_path)
    flags[0] = "false"
    result = _run(["bash", str(TRAIN_THEN_EVAL), *flags], {"EVAL_CHAIN_CMD": stub})
    assert result.returncode != 0
    assert not out.exists()


def _stub_python(tmp_path: Path) -> tuple[Path, Path]:
    """A stub python on PATH that logs its GPU and module, one line per
    call."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.txt"
    stub = bin_dir / "python"
    stub.write_text(f'#!/usr/bin/env bash\necho "$CUDA_VISIBLE_DEVICES $2" >> {calls}\n')
    stub.chmod(0o755)
    return bin_dir, calls


def _chain_env(tmp_path: Path, bin_dir: Path, **extra: str) -> dict[str, str]:
    return {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "EVAL_RUN": "run",
        "EVAL_DIR": str(tmp_path / "eval"),
        "EVAL_HF_BASE": "Qwen/Qwen3-0.6B-Base",
        "EVAL_OBJECTIVE": "bd3",
        "EVAL_DATA_PREFIX": "/data/prefix",
        **extra,
    }


def test_eval_chain_runs_m1_and_each_mode_in_order(tmp_path: Path) -> None:
    bin_dir, calls = _stub_python(tmp_path)
    env = _chain_env(tmp_path, bin_dir, CUDA_VISIBLE_DEVICES="3")
    result = _run(["bash", str(EVAL_CHAIN)], env)
    assert result.returncode == 0, result.stderr
    assert calls.read_text().splitlines() == [
        "3 oellm_autoexp.dllm.eval.eval_nelbo",
        "3 oellm_autoexp.dllm.eval.eval_gsm8k",
        "3 oellm_autoexp.dllm.eval.eval_gsm8k",
    ]


def test_eval_chain_parallel_puts_each_task_on_its_own_gpu(tmp_path: Path) -> None:
    bin_dir, calls = _stub_python(tmp_path)
    env = _chain_env(tmp_path, bin_dir, CUDA_VISIBLE_DEVICES="4,5,6,7", EVAL_PARALLEL="1")
    result = _run(["bash", str(EVAL_CHAIN)], env)
    assert result.returncode == 0, result.stderr
    assert sorted(calls.read_text().splitlines()) == [
        "4 oellm_autoexp.dllm.eval.eval_nelbo",
        "5 oellm_autoexp.dllm.eval.eval_gsm8k",
        "6 oellm_autoexp.dllm.eval.eval_gsm8k",
    ]
    for name in ("m1", "gsm8k_static", "gsm8k_dynamic"):
        assert (tmp_path / "eval" / "logs" / f"run.{name}.log").exists()


def test_eval_chain_reports_a_failed_parallel_task(tmp_path: Path) -> None:
    bin_dir, _ = _stub_python(tmp_path)
    (bin_dir / "python").write_text(
        '#!/usr/bin/env bash\n[ "$2" != "oellm_autoexp.dllm.eval.eval_nelbo" ]\n'
    )
    env = _chain_env(tmp_path, bin_dir, CUDA_VISIBLE_DEVICES="0,1,2", EVAL_PARALLEL="1")
    result = _run(["bash", str(EVAL_CHAIN)], env)
    assert result.returncode == 1
    assert "m1 FAILED" in result.stdout
