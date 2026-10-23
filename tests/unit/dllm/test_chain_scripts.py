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

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_THEN_EVAL = REPO_ROOT / "oellm_autoexp/dllm/train_then_eval.sh"
EVAL_CHAIN = REPO_ROOT / "oellm_autoexp/dllm/eval/eval_chain.sh"
_EVAL_KEYS = (
    "EVAL_CKPT",
    "EVAL_OBJECTIVE",
    "EVAL_BLOCK_SIZE",
    "EVAL_TOKEN_SHIFT",
    "EVAL_MASK_ID",
    "EVAL_DOC_MASK",
    "EVAL_CLEAN_ATTENTION",
    "EVAL_M1_AR",
)


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
        "EVAL_DOC_MASK": "1",
        "EVAL_CLEAN_ATTENTION": "",
        "EVAL_M1_AR": "",
        "EVAL_MODES": "",
    }


def test_evaluation_keeps_the_runs_document_mask_and_reads_none_in_any_case(
    tmp_path: Path,
) -> None:
    """R09 trains with --bd-doc-mask 0, so its M1 must use the same attention
    rule."""
    save = tmp_path / "checkpoints"
    save.mkdir()
    (save / "latest_checkpointed_iteration.txt").write_text("100")
    stub, out = _stub_eval(tmp_path)
    flags = _train_flags(save, "--bd-doc-mask", "0")
    flags[flags.index("--bd-block-schedule") + 1] = " NONE "
    result = _run(["bash", str(TRAIN_THEN_EVAL), *flags], {"EVAL_CHAIN_CMD": stub})
    assert result.returncode == 0, result.stderr
    settings = _settings(out)
    assert (settings["EVAL_DOC_MASK"], settings["EVAL_BLOCK_SIZE"]) == ("0", "8")


def test_missing_save_or_train_iters_is_reported(tmp_path: Path) -> None:
    stub, out = _stub_eval(tmp_path)
    flags = _train_flags(tmp_path)
    flags.remove("--train-iters")
    flags.remove("100")
    result = _run(["bash", str(TRAIN_THEN_EVAL), *flags], {"EVAL_CHAIN_CMD": stub})
    assert result.returncode == 0, result.stderr
    assert "WARNING" in result.stdout
    assert not out.exists()


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


STAGE_DATA = REPO_ROOT / "oellm_autoexp/dllm/stage_data.sh"


def _dataset(tmp_path: Path, *names: str) -> list[str]:
    data = tmp_path / "data"
    data.mkdir()
    prefixes = []
    for name in names:
        (data / f"{name}.bin").write_bytes(b"\x01" * 4096)
        (data / f"{name}.idx").write_bytes(b"\x02" * 64)
        prefixes.append(str(data / name))
    return prefixes


def _recording_command(tmp_path: Path) -> tuple[Path, Path]:
    """A command that writes each argument, and whether <arg>.bin exists, to a
    file."""
    record = tmp_path / "args.txt"
    script = tmp_path / "record.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        f'for a in "$@"; do if [ -f "$a.bin" ]; then echo "$a bin"; else echo "$a"; fi; done > {record}\n'
        'exit "${RECORD_EXIT:-0}"\n'
    )
    script.chmod(0o755)
    return script, record


def test_stage_data_copies_rewrites_and_deletes_the_local_copy(tmp_path: Path) -> None:
    prefix_a, prefix_b = _dataset(tmp_path, "shard_00_text_document", "shard_15_text_document")
    command, record = _recording_command(tmp_path)
    local = tmp_path / "local"
    local.mkdir()
    args = [str(command), "--train-data-path", "1.0", prefix_a, "--valid-data-path", prefix_b,
            "--data-cache-path", str(tmp_path / "cache")]  # fmt: skip
    result = _run(["bash", str(STAGE_DATA), *args], {"DLLM_LOCAL_SCRATCH": str(local)})
    assert result.returncode == 0, result.stderr
    lines = record.read_text().splitlines()
    staged = [line.removesuffix(" bin") for line in lines if line.endswith(" bin")]
    assert len(staged) == 2
    assert all(path.startswith(f"{local}/dllm-data/") for path in staged)
    assert {Path(path).name.split("_", 1)[1] for path in staged} == {
        "shard_00_text_document",
        "shard_15_text_document",
    }
    assert lines[:2] == ["--train-data-path", "1.0"]
    assert lines[2].endswith("_shard_00_text_document bin")
    assert lines[3] == "--valid-data-path"
    assert lines[4].endswith("_shard_15_text_document bin")
    assert lines[5:] == ["--data-cache-path", str(tmp_path / "cache")]
    assert not (local / "dllm-data").exists()


def test_stage_data_keeps_the_paths_without_local_scratch(tmp_path: Path) -> None:
    (prefix,) = _dataset(tmp_path, "shard_00_text_document")
    command, record = _recording_command(tmp_path)
    env = {"TMPDIR": str(tmp_path), "SLURM_JOB_ID": "0"}
    result = _run(["bash", str(STAGE_DATA), str(command), "--train-data-path", prefix], env)
    assert result.returncode == 0, result.stderr
    assert record.read_text().splitlines() == ["--train-data-path", f"{prefix} bin"]


def test_stage_data_deletes_the_copy_after_a_failed_command(tmp_path: Path) -> None:
    (prefix,) = _dataset(tmp_path, "shard_00_text_document")
    command, _ = _recording_command(tmp_path)
    local = tmp_path / "local"
    local.mkdir()
    env = {"DLLM_LOCAL_SCRATCH": str(local), "RECORD_EXIT": "3"}
    result = _run(["bash", str(STAGE_DATA), str(command), "--train-data-path", prefix], env)
    assert result.returncode == 3
    assert not (local / "dllm-data").exists()


def test_eval_chain_exports_the_checkpoint_and_names_the_ar_result(tmp_path: Path) -> None:
    """The export gets the run's settings, M1 its document mask, and mode ar
    its result name."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.txt"
    (bin_dir / "python").write_text(f'#!/usr/bin/env bash\necho "$*" >> {calls}\n')
    (bin_dir / "python").chmod(0o755)
    env = _chain_env(
        tmp_path,
        bin_dir,
        EVAL_CKPT="/ckpt",
        EVAL_BLOCK_SIZE="1",
        EVAL_TOKEN_SHIFT="1",
        EVAL_MASK_ID="151669",
        EVAL_DOC_MASK="0",
        EVAL_MODES="ar",
    )
    result = _run(["bash", str(EVAL_CHAIN)], env)
    assert result.returncode == 0, result.stderr
    export, m1, gsm8k = calls.read_text().splitlines()
    model = f"{tmp_path}/eval/hf/run"
    assert export.split() == [
        "-m",
        "oellm_autoexp.dllm.convert.megadlm_to_hf_qwen3",
        "--megatron",
        "/ckpt",
        "--hf-base",
        "Qwen/Qwen3-0.6B-Base",
        "--out",
        model,
        "--block-size",
        "1",
        "--token-shift",
        "1",
        "--mask-token-id",
        "151669",
        "--layout",
        "te",
    ]  # fmt: skip  (no --clean-attention: the exporter takes it from bd_objective)
    assert "--doc-mask 0" in m1 and f"--model {model}" in m1
    assert gsm8k.endswith(f"--out {tmp_path}/eval/run.gsm8k_ar.json")


def test_eval_chain_passes_a_set_clean_attention_to_the_export(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.txt"
    (bin_dir / "python").write_text(f'#!/usr/bin/env bash\necho "$*" >> {calls}\n')
    (bin_dir / "python").chmod(0o755)
    env = _chain_env(
        tmp_path,
        bin_dir,
        EVAL_CKPT="/ckpt",
        EVAL_BLOCK_SIZE="32",
        EVAL_TOKEN_SHIFT="0",
        EVAL_MASK_ID="151669",
        EVAL_CLEAN_ATTENTION="causal",
        EVAL_NUM_SEQS="0",
        EVAL_MODES="none",
    )
    result = _run(["bash", str(EVAL_CHAIN)], env)
    assert result.returncode == 0, result.stderr
    (export,) = calls.read_text().splitlines()
    assert export.endswith("--layout te --clean-attention causal")


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"EVAL_DTYPE": "float16"}, "EVAL_DTYPE"),
        ({"EVAL_NUM_SEQS": "many"}, "EVAL_NUM_SEQS"),
        ({"EVAL_PARALLEL": "1"}, "found no GPU"),
    ],
)
def test_eval_chain_fails_on_settings_that_would_skip_or_break_tasks(
    tmp_path: Path, extra: dict[str, str], message: str
) -> None:
    bin_dir, calls = _stub_python(tmp_path)
    (bin_dir / "nvidia-smi").write_text("#!/usr/bin/env bash\nexit 0\n")  # lists no GPU
    (bin_dir / "nvidia-smi").chmod(0o755)
    env = _chain_env(tmp_path, bin_dir, CUDA_VISIBLE_DEVICES="", **extra)  # none visible
    result = _run(["bash", str(EVAL_CHAIN)], env)
    assert result.returncode != 0
    assert message in result.stderr
    assert not calls.exists()


def test_stage_data_needs_a_command(tmp_path: Path) -> None:
    result = _run(["bash", str(STAGE_DATA)], {})
    assert result.returncode == 64
    assert "usage" in result.stderr


def test_stage_data_falls_back_when_the_override_directory_is_missing(tmp_path: Path) -> None:
    (prefix,) = _dataset(tmp_path, "shard_00_text_document")
    command, record = _recording_command(tmp_path)
    env = {"DLLM_LOCAL_SCRATCH": str(tmp_path / "absent")}
    result = _run(["bash", str(STAGE_DATA), str(command), "--train-data-path", prefix], env)
    assert result.returncode == 0, result.stderr
    assert record.read_text().splitlines() == ["--train-data-path", f"{prefix} bin"]


def test_nld_run_is_evaluated_as_diffusion_with_causal_clean_stream_and_ar_mode(
    tmp_path: Path,
) -> None:
    """Spec section 15: M1 is the bd3 NELBO with the nld mask, plus the AR-mode exact NLL."""
    save = tmp_path / "checkpoints"
    save.mkdir()
    (save / "latest_checkpointed_iteration.txt").write_text("100")
    stub, out = _stub_eval(tmp_path)
    flags = _train_flags(save)
    flags[flags.index("--bd-objective") + 1] = "nld"
    flags[flags.index("--bd-block-size") + 1] = "32"
    env = {"EVAL_CHAIN_CMD": stub, "EVAL_MODES": "static,dynamic"}
    result = _run(["bash", str(TRAIN_THEN_EVAL), *flags], env)
    assert result.returncode == 0, result.stderr
    settings = _settings(out)
    assert settings["EVAL_OBJECTIVE"] == "bd3"
    assert (settings["EVAL_BLOCK_SIZE"], settings["EVAL_TOKEN_SHIFT"]) == ("32", "0")
    assert (settings["EVAL_CLEAN_ATTENTION"], settings["EVAL_M1_AR"]) == ("causal", "1")
    assert settings["EVAL_MODES"] == "static,dynamic"


@pytest.mark.parametrize(("dtype", "expected"), [(None, "bfloat16"), ("float32", "float32")])
def test_eval_chain_passes_the_weights_dtype_to_every_task(
    tmp_path: Path, dtype: str | None, expected: str
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.txt"
    (bin_dir / "python").write_text(f'#!/usr/bin/env bash\necho "$*" >> {calls}\n')
    (bin_dir / "python").chmod(0o755)
    extra = {"EVAL_M1_AR": "1", "EVAL_MODES": "ar,static"}
    if dtype is not None:
        extra["EVAL_DTYPE"] = dtype
    result = _run(["bash", str(EVAL_CHAIN)], _chain_env(tmp_path, bin_dir, **extra))
    assert result.returncode == 0, result.stderr
    lines = calls.read_text().splitlines()
    assert len(lines) == 4  # m1, m1_ar, gsm8k_ar, gsm8k_static
    assert all(f"--dtype {expected} " in line for line in lines)


def test_eval_chain_adds_the_ar_mode_m1_when_asked(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.txt"
    (bin_dir / "python").write_text(f'#!/usr/bin/env bash\necho "$*" >> {calls}\n')
    (bin_dir / "python").chmod(0o755)
    env = _chain_env(tmp_path, bin_dir, EVAL_M1_AR="1", EVAL_MODES="none")
    result = _run(["bash", str(EVAL_CHAIN)], env)
    assert result.returncode == 0, result.stderr
    m1, m1_ar = calls.read_text().splitlines()
    assert "--objective bd3" in m1 and m1.endswith(f"{tmp_path}/eval/run.m1.json")
    assert "--objective ar" in m1_ar and m1_ar.endswith(f"{tmp_path}/eval/run.m1_ar.json")
