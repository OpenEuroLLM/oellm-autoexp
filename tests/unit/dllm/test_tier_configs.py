"""Every tier 0 and tier 1 job runs the configuration that its run name states.

Run names (.claude/dllm/ABLATIONS.md) key the RESULTS.md tables through collect_results.py, so a
name that disagrees with the job's flags would report a result under the wrong factor. The jobs
are rendered as ``scripts/run_autoexp.py --dry-run`` does; nothing is submitted.

Run with:
    pytest tests/unit/dllm/test_tier_configs.py -v
"""

from __future__ import annotations

import math
import os
import re
import shlex
from pathlib import Path

import pytest

from oellm_autoexp.dllm.eval.collect_results import parse_run_name
from oellm_autoexp.dllm.render import render_launch_commands

REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = REPO_ROOT / "config"
_TIER_CONFIGS = (
    "experiments/dllm/bd/tier0_qwen3_0.6B",
    "experiments/dllm/bd/tier1_qwen3_0.6B",
)
_BASE_GLOBAL_BATCH = 256  # PARAMS training.global_batch_size
_L = 2048  # PARAMS training.seq_length_effective
_DUMMY_ENVS = {
    "SLURM_ACCOUNT": "test_account",
    "HF_HOME": "/dummy/hf_home",
    "DATA_DIR": "/dummy/data",
    "OELLM_DATASETS_TOKENIZED_DIR": "/dummy/datasets",
    "CONTAINER_CACHE_DIR": "/dummy/containers",
    "WORK": "/dummy/work",
    "TORCH_HOME": "/dummy/torch_home",
}


def _flags(command: str) -> dict[str, str]:
    """Map each ``--flag`` of a launch command to its last value ("" for a
    switch)."""
    tokens = shlex.split(re.sub(r"\\\r?\n", " ", command))
    values: dict[str, str] = {}
    for i, token in enumerate(tokens):
        if token.startswith("--"):
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            values[token] = "" if nxt.startswith("--") else nxt
    return values


def _jobs(config_name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict]:
    for key, value in _DUMMY_ENVS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("PWD", str(REPO_ROOT))
    if "USER" not in os.environ:
        monkeypatch.setenv("USER", "dummy_user")
    return [_flags(command) for command in render_launch_commands(config_name, _CONFIG_DIR)]


@pytest.mark.parametrize("config_name", _TIER_CONFIGS, ids=lambda name: name.split("/")[-1])
def test_tier_job_flags_match_their_run_names(
    config_name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    jobs = _jobs(config_name, monkeypatch, tmp_path)
    assert jobs, f"{config_name}: rendered no jobs"
    names = [job["--wandb-exp-name"] for job in jobs]
    assert len(set(names)) == len(names), f"{config_name}: duplicate run names {names}"
    for flags in jobs:
        name = flags["--wandb-exp-name"]
        run = parse_run_name(name)
        assert flags["--bd-objective"] == run["objective"], name
        assert float(flags["--lr"]) == pytest.approx(run["lr"]), name
        assert int(flags["--seq-length"]) == _L + 1, name
        iters = math.ceil(run["tokens"] / (_BASE_GLOBAL_BATCH * _L))
        assert int(flags["--train-iters"]) == iters, name
        # PARAMS training: warmup max(min_warmup_iters 50, 1 percent); per-token loss (spec 7).
        assert int(flags["--lr-warmup-iters"]) == max(50, math.ceil(0.01 * iters)), name
        assert "--calculate-per-token-loss" in flags, name
        if run["objective"] == "ar":
            continue
        assert int(flags["--bd-block-size"]) == run["block_size"], name
        assert int(flags["--bd-token-shift"]) == run["token_shift"], name
        assert int(flags["--bd-complementary-mask"]) == run["complementary"], name
        assert flags["--bd-loss-weighting"] == run["weighting"], name
        assert float(flags["--bd-t-min"]) == run["t_min"], name
        assert float(flags["--bd-t-max"]) == run["t_max"], name
        assert int(flags["--bd-doc-mask"]) == run["doc_mask"], name
        expected_schedule = "none" if run["schedule"] is None else run["schedule"]
        assert flags["--bd-block-schedule"] == expected_schedule, name


def test_complementary_run_halves_the_batch_of_its_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R05 matches R00's FLOPs per step: half the global and micro batch (spec
    section 3)."""
    jobs = _jobs("experiments/dllm/bd/tier1_qwen3_0.6B", monkeypatch, tmp_path)
    plain = [flags for flags in jobs if flags.get("--bd-complementary-mask") == "0"]
    comp = [flags for flags in jobs if flags.get("--bd-complementary-mask") == "1"]
    assert plain and comp
    for flags in plain:
        assert int(flags["--global-batch-size"]) == _BASE_GLOBAL_BATCH
    micro = {int(flags["--micro-batch-size"]) for flags in plain}
    assert len(micro) == 1
    (base_micro,) = micro
    for flags in comp:
        assert int(flags["--global-batch-size"]) == _BASE_GLOBAL_BATCH // 2
        assert int(flags["--micro-batch-size"]) == base_micro // 2
