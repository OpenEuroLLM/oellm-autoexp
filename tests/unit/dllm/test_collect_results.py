"""Tests for oellm_autoexp.dllm.eval.collect_results (no torch needed)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from oellm_autoexp.dllm.eval import collect_results as cr

BD3_NAME = "bd3-q3_0.6B-B8-sh0-cm0-welbo-t0_1-dm1-scnone-lr3e-5-T2B"
BD3_B4_NAME = "bd3-q3_0.6B-B4-sh0-cm0-welbo-t0_1-dm1-scnone-lr3e-5-T2B"
AR_NAME = "ar-q3_0.6B-lr3e-5-T2B"
NLD_NAME = "nld-q3_0.6B-B32-sh0-a0.3-ao0.2-lr3e-5-T2B"
NLD_TIER1_ROW = (
    f"| {NLD_NAME} | objective nld (B = 32, diffusion weight 0.3, AR-only 0.2) "
    "| 1.1500 | -4.17 | - | - | - | - | AR-mode M1 1.1200 |"
)
T0_RUNS = {
    "1e-5": "bd3-q3_0.6B-B8-sh0-cm0-welbo-t0_1-dm1-scnone-lr1e-5-T0.5B",
    "3e-5": "bd3-q3_0.6B-B8-sh0-cm0-welbo-t0_1-dm1-scnone-lr3e-5-T0.5B",
    "1e-4": "bd3-q3_0.6B-B8-sh0-cm0-welbo-t0_1-dm1-scnone-lr1e-4-T0.5B",
}
T0_M1 = {"1e-5": 1.30, "3e-5": 1.20, "1e-4": 1.25}
SEP_TIER0 = "|---|---|---|---|"
HEADER_TIER0 = "| Run | LR | M1 | Notes |"
HEADER_TIER1 = "| Run | Change | M1 | dM1 vs R00 (%) | M2 static | M2 dyn@0.9 | M3 | M4 | Notes |"
SEP_TIER1 = "|---|---|---|---|---|---|---|---|---|"

RESULTS_LINES = [
    "# Results",
    "",
    (
        "M1 = offline validation NELBO bits/token (AR rows: exact NLL). "
        "M2 = GSM8K accuracy (percent)."
    ),
    "M3 = TPF_decode at dynamic@0.9. M4 = training tokens/s/GPU.",
    "",
    "## References (no training)",
    "| Model | M1 | M2 (AR greedy) |",
    "|---|---|---|",
    "",
    "## Tier 0 (0.6B, 0.5B tokens)",
    HEADER_TIER0,
    SEP_TIER0,
    "",
    "## Tier 1 (0.6B, 2B tokens)",
    HEADER_TIER1,
    SEP_TIER1,
    "",
    "## Tier 2",
    "| Item | Run | Size | Tokens | M1 | M2 static | M2 dyn@0.9 | M3 | Notes |",
    "|---|---|---|---|---|---|---|---|---|",
    "",
    "## Selection log",
    "<!-- one line per applied rule, with the D-entry it produced -->",
]
RESULTS_MD = "\n".join(RESULTS_LINES) + "\n"

EXPECTED_TIER0 = [
    f"| {T0_RUNS['1e-5']} | 1e-5 | 1.3000 |  |",
    f"| {T0_RUNS['3e-5']} | 3e-5 | 1.2000 | picked |",
    f"| {T0_RUNS['1e-4']} | 1e-4 | 1.2500 |  |",
]
EXPECTED_TIER1 = [
    f"| {BD3_NAME} | none | 1.2000 | +0.00 | 30.5 | 29.0 | 2.40 | 400 | base |",
    f"| {AR_NAME} | objective ar | 1.1000 | -8.33 | 33.2 | - | - | - | M2 = AR greedy |",
    f"| {BD3_B4_NAME} | B = 4 | 1.2030 | +0.25 | 31.0 | 30.0 | 2.50 | - | tie |",
]


def _write_json(path: Path, payload: dict[str, float]) -> None:
    """Write a JSON object to a file."""
    path.write_text(json.dumps(payload), encoding="utf-8")


def _throughput_log(values: Sequence[float]) -> str:
    """Build a synthetic training log with one ``bd throughput`` line per
    value."""
    lines = [" iteration        1 | lm loss: 2.0 |"]
    for iteration, value in enumerate(values, start=1):
        lines.append(f" bd throughput | iteration {iteration:8d} | tokens/s/GPU: {value:.1f} |")
    return "\n".join(lines) + "\n"


def _cells(line: str) -> list[str]:
    """Split one markdown table row into stripped cells."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _tier0_rows(*pairs: tuple[float, float]) -> list[dict[str, Any]]:
    """Build tier 0 rows from (LR, M1) pairs."""
    return [{"run": f"run-{index}", "lr": lr, "m1": m1} for index, (lr, m1) in enumerate(pairs)]


def _lines_outside_tier_tables(text: str) -> list[str]:
    """Return every line except the table rows under the Tier 0 and Tier 1
    headings."""
    kept: list[str] = []
    in_tier_section = False
    for line in text.splitlines():
        if line.startswith("## "):
            in_tier_section = line.startswith(("## Tier 0", "## Tier 1"))
        elif in_tier_section and line.startswith("|"):
            continue
        kept.append(line)
    return kept


def _write_eval_dir(root: Path) -> Path:
    """Create an evaluation directory with tier 0 and tier 1 runs and one
    reference."""
    eval_dir = root / "eval"
    eval_dir.mkdir()
    for key, run in T0_RUNS.items():
        _write_json(eval_dir / f"{run}.m1.json", {"bits_per_token": T0_M1[key]})
    _write_json(eval_dir / f"{BD3_NAME}.m1.json", {"bits_per_token": 1.2})
    _write_json(eval_dir / f"{BD3_NAME}.gsm8k_static.json", {"accuracy": 30.5})
    _write_json(
        eval_dir / f"{BD3_NAME}.gsm8k_dynamic.json",
        {"accuracy": 29.0, "mean_tpf_decode": 2.4},
    )
    _write_json(eval_dir / f"{BD3_B4_NAME}.m1.json", {"bits_per_token": 1.203})
    _write_json(eval_dir / f"{BD3_B4_NAME}.gsm8k_static.json", {"accuracy": 31.0})
    _write_json(
        eval_dir / f"{BD3_B4_NAME}.gsm8k_dynamic.json",
        {"accuracy": 30.0, "mean_tpf_decode": 2.5},
    )
    _write_json(eval_dir / f"{AR_NAME}.m1.json", {"bits_per_token": 1.1})
    _write_json(eval_dir / f"{AR_NAME}.gsm8k_ar.json", {"accuracy": 33.2})
    _write_json(eval_dir / "qwen3-0.6B-base.m1.json", {"bits_per_token": 1.0})
    return eval_dir


def _write_logs(root: Path) -> Path:
    """Create a logs directory with one training log of the R00 run."""
    logs = root / "logs"
    logs.mkdir()
    log_text = _throughput_log([100.0, 200.0, 300.0, 400.0, 500.0])
    (logs / f"{BD3_NAME}_12345.out").write_text(log_text, encoding="utf-8")
    return logs


@pytest.mark.parametrize(
    "name",
    [
        "",
        "qwen3-0.6B-base",
        "bd3-q3_0.6B-B8",
        "bd3-q3_0.6B-B8-sh2-cm0-welbo-t0_1-dm1-scnone-lr3e-5-T2B",
        "bd3-q3_0.6B-B8-sh0-cm0-welbo-t0.8_0.2-dm1-scnone-lr3e-5-T2B",
        "bd3-q3_0.6B-B0-sh0-cm0-welbo-t0_1-dm1-scnone-lr3e-5-T2B",
        "ar-q3_0.6B-lr3e-5",
        "nld-q3_0.6B-B32-sh0-a0.3-lr3e-5-T2B",
        "nld-q3_0.6B-B32-sh0-a0.3-ao1.5-lr3e-5-T2B",
        "nld-q3_0.6B-B32-sh0-a0.3-ao1-lr3e-5-T2B",  # args.py needs a fraction below 1
        "nld-q3_0.6B-B0-sh0-a0.3-ao0.2-lr3e-5-T2B",
    ],
)
def test_parse_run_name_rejects_bad_names(name: str) -> None:
    with pytest.raises(ValueError):
        cr.parse_run_name(name)


def test_parse_bd3_run_name_from_ablation_pattern() -> None:
    assert cr.parse_run_name(BD3_NAME) == {
        "objective": "bd3",
        "size": "0.6B",
        "block_size": 8,
        "token_shift": 0,
        "complementary": 0,
        "weighting": "elbo",
        "t_min": 0.0,
        "t_max": 1.0,
        "doc_mask": 1,
        "schedule": None,
        "diffusion_weight": None,
        "ar_only_fraction": None,
        "lr": 3e-5,
        "tokens": 2_000_000_000,
    }


def test_parse_bd3_run_name_with_schedule_and_uniform_weighting() -> None:
    name = "bd3-q3_0.6B-B8-sh1-cm1-wuniform-t0.2_0.8-dm0-sc0:4,0.15:8-lr1e-4-T0.5B"
    parsed = cr.parse_run_name(name)
    assert parsed["token_shift"] == 1
    assert parsed["complementary"] == 1
    assert parsed["weighting"] == "uniform"
    assert (parsed["t_min"], parsed["t_max"]) == (0.2, 0.8)
    assert parsed["doc_mask"] == 0
    assert parsed["schedule"] == "0:4,0.15:8"
    assert parsed["lr"] == 1e-4
    assert parsed["tokens"] == 500_000_000


def test_parse_bd3_run_name_with_path_safe_schedule() -> None:
    """The spelling 0-4_0.15-8 (no ':' or ',' in paths) parses to the flag text
    0:4,0.15:8."""
    name = "bd3-q3_0.6B-B8-sh0-cm0-welbo-t0_1-dm1-sc0-4_0.15-8-lr3e-5-T2B"
    parsed = cr.parse_run_name(name)
    assert parsed["schedule"] == "0:4,0.15:8"
    assert parsed["block_size"] == 8
    assert parsed["lr"] == 3e-5
    assert parsed["tokens"] == 2_000_000_000


def test_parse_nld_run_name_keeps_the_r00_base_and_adds_the_nld_settings() -> None:
    assert cr.parse_run_name(NLD_NAME) == {
        "objective": "nld",
        "size": "0.6B",
        "block_size": 32,
        "token_shift": 0,
        "complementary": 0,
        "weighting": "elbo",
        "t_min": 0.0,
        "t_max": 1.0,
        "doc_mask": 1,
        "schedule": None,
        "diffusion_weight": 0.3,
        "ar_only_fraction": 0.2,
        "lr": 3e-5,
        "tokens": 2_000_000_000,
    }


def test_parse_nld_run_name_reads_token_shift_and_the_boundary_fractions() -> None:
    parsed = cr.parse_run_name("nld-q3_0.6B-B8-sh1-a1-ao0-lr1e-4-T0.5B")
    assert parsed["token_shift"] == 1
    assert parsed["block_size"] == 8
    assert parsed["diffusion_weight"] == 1.0
    assert parsed["ar_only_fraction"] == 0.0
    assert parsed["lr"] == 1e-4
    assert parsed["tokens"] == 500_000_000


def test_parse_ar_run_name_has_no_block_diffusion_fields() -> None:
    parsed = cr.parse_run_name(AR_NAME)
    assert parsed["objective"] == "ar"
    assert parsed["size"] == "0.6B"
    assert parsed["lr"] == 3e-5
    assert parsed["tokens"] == 2_000_000_000
    bd_fields = (
        "block_size",
        "token_shift",
        "complementary",
        "weighting",
        "t_min",
        "t_max",
        "doc_mask",
        "schedule",
        "diffusion_weight",
        "ar_only_fraction",
    )
    assert all(parsed[key] is None for key in bd_fields)


@pytest.mark.parametrize(("lr", "text"), [(1e-5, "1e-5"), (3e-4, "3e-4"), (2.5e-5, "2.5e-5")])
def test_format_lr_is_compact(lr: float, text: str) -> None:
    assert cr.format_lr(lr) == text


def test_tokens_per_s_gpu_is_median_after_skipping_warmup() -> None:
    log = _throughput_log([100.0, 200.0, 300.0, 400.0, 900.0])
    assert cr.tokens_per_s_gpu(log) == 400.0
    assert cr.tokens_per_s_gpu(log, skip_first=0) == 300.0
    assert cr.tokens_per_s_gpu(log, skip_first=3) == 650.0  # median of [400, 900]
    assert cr.tokens_per_s_gpu(log, skip_first=4) == 900.0
    assert cr.tokens_per_s_gpu(_throughput_log([100.0, 200.0, 300.0, 400.0])) == 350.0


def test_tokens_per_s_gpu_is_none_when_absent_or_fully_skipped() -> None:
    assert cr.tokens_per_s_gpu("loss 1.0\n") is None
    assert cr.tokens_per_s_gpu(_throughput_log([100.0, 200.0]), skip_first=2) is None
    with pytest.raises(ValueError):
        cr.tokens_per_s_gpu(_throughput_log([100.0]), skip_first=-1)


def test_load_eval_dir_reads_each_metric_file(tmp_path: Path) -> None:
    _write_json(tmp_path / f"{BD3_NAME}.m1.json", {"bits_per_token": 1.2})
    _write_json(tmp_path / f"{BD3_NAME}.gsm8k_static.json", {"accuracy": 30.5})
    _write_json(
        tmp_path / f"{BD3_NAME}.gsm8k_dynamic.json",
        {"accuracy": 29.0, "mean_tpf_decode": 2.4},
    )
    _write_json(tmp_path / f"{AR_NAME}.m1.json", {"bits_per_token": 1.1})
    _write_json(tmp_path / f"{AR_NAME}.gsm8k_ar.json", {"accuracy": 33.2})
    _write_json(tmp_path / "parity.json", {"top1": 0.99})
    assert cr.load_eval_dir(tmp_path) == {
        BD3_NAME: {
            "m1": 1.2,
            "m2_static": 30.5,
            "m2_dyn": 29.0,
            "m3": 2.4,
            "m2_ar": None,
            "m1_ar": None,
        },
        AR_NAME: {
            "m1": 1.1,
            "m2_static": None,
            "m2_dyn": None,
            "m3": None,
            "m2_ar": 33.2,
            "m1_ar": None,
        },
    }


def test_load_eval_dir_reads_the_ar_mode_m1_of_an_nld_run(tmp_path: Path) -> None:
    _write_json(tmp_path / f"{NLD_NAME}.m1.json", {"bits_per_token": 1.15})
    _write_json(tmp_path / f"{NLD_NAME}.m1_ar.json", {"bits_per_token": 1.12})
    assert cr.load_eval_dir(tmp_path) == {
        NLD_NAME: {
            "m1": 1.15,
            "m2_static": None,
            "m2_dyn": None,
            "m3": None,
            "m2_ar": None,
            "m1_ar": 1.12,
        },
    }


def test_load_eval_dir_rejects_a_missing_field(tmp_path: Path) -> None:
    _write_json(tmp_path / f"{BD3_NAME}.m1.json", {"nelbo": 1.2})
    with pytest.raises(ValueError, match="bits_per_token"):
        cr.load_eval_dir(tmp_path)


def test_load_eval_dir_rejects_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cr.load_eval_dir(tmp_path / "absent")


def test_collect_throughput_matches_logs_by_run_name(tmp_path: Path) -> None:
    logs = _write_logs(tmp_path)
    (logs / "unrelated_999.out").write_text(_throughput_log([1.0, 2.0, 3.0]), encoding="utf-8")
    result = cr.collect_throughput(str(logs / "*.out"), [BD3_NAME, AR_NAME])
    assert result == {BD3_NAME: 400.0, AR_NAME: None}


def test_collect_throughput_reads_logs_in_run_directories(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The tier jobs write runs/<run>/slurm-<id>.log, so the directory names
    the run."""
    for run, values in ((BD3_NAME, [1.0, 2.0, 30.0, 40.0, 50.0]), (AR_NAME, [9.0, 9.0, 7.0])):
        run_dir = tmp_path / "runs" / run
        run_dir.mkdir(parents=True)
        (run_dir / "slurm-28458579.log").write_text(_throughput_log(values), encoding="utf-8")
    stray = tmp_path / "runs" / "probe_x"
    stray.mkdir()
    (stray / "slurm-1.log").write_text(_throughput_log([5.0, 5.0, 5.0]), encoding="utf-8")
    result = cr.collect_throughput(str(tmp_path / "runs/*/slurm-*.log"), [BD3_NAME, AR_NAME])
    assert result == {BD3_NAME: 40.0, AR_NAME: 7.0}
    assert "probe_x" in capsys.readouterr().err


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_load_eval_dir_rejects_a_non_finite_metric(tmp_path: Path, value: float) -> None:
    _write_json(tmp_path / f"{BD3_NAME}.m1.json", {"bits_per_token": value})
    with pytest.raises(ValueError, match="not finite"):
        cr.load_eval_dir(tmp_path)


def test_main_write_refuses_an_empty_collection(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    results_md = tmp_path / "RESULTS.md"
    results_md.write_text(RESULTS_MD, encoding="utf-8")
    argv = ["--eval-dir", str(eval_dir), "--results-md", str(results_md), "--write"]
    assert cr.main(argv) == 1
    assert results_md.read_text(encoding="utf-8") == RESULTS_MD


def test_tier1_table_reports_dm1_ties_and_missing_values() -> None:
    rows = [
        {
            "run": BD3_NAME,
            "m1": 1.2,
            "m2_static": 30.0,
            "m2_dyn": 29.5,
            "m3": 2.1,
            "m4": 12345.6,
        },
        {"run": BD3_B4_NAME, "m1": 1.203, "m2_static": 31.04, "m2_dyn": 30.0, "m3": 2.5},
        {"run": AR_NAME, "m1": None},
    ]
    lines = cr.tier1_table(rows, base_run=BD3_NAME).splitlines()
    assert lines[0] == HEADER_TIER1
    assert lines[1] == SEP_TIER1
    assert lines[2] == (
        f"| {BD3_NAME} | none | 1.2000 | +0.00 | 30.0 | 29.5 | 2.10 | 12346 | base |"
    )
    assert lines[3] == (
        f"| {BD3_B4_NAME} | B = 4 | 1.2030 | +0.25 | 31.0 | 30.0 | 2.50 | - | tie |"
    )
    assert lines[4] == f"| {AR_NAME} | objective ar | - | - | - | - | - | - | M2 = AR greedy |"


def test_tier1_change_column_lists_every_changed_factor() -> None:
    name = "bd3-q3_0.6B-B8-sh1-cm1-wuniform-t0.2_0.8-dm0-sc0:4,0.15:8-lr1e-4-T2B"
    lines = cr.tier1_table([{"run": name, "m1": 1.3}], base_run=BD3_NAME).splitlines()
    cells = _cells(lines[2])
    assert cells[1] == (
        "token shift 1; complementary mask 1; uniform weighting; t in [0.2, 0.8]; "
        "doc mask 0; schedule 0:4,0.15:8; lr 1e-4"
    )
    assert cells[3] == "-"


def test_tier1_row_of_an_nld_run_shows_its_change_and_ar_mode_m1() -> None:
    rows = [
        {"run": BD3_NAME, "m1": 1.2, "m2_static": 30.5, "m2_dyn": 29.0, "m3": 2.4, "m4": 400.0},
        {"run": NLD_NAME, "m1": 1.15, "m2_static": 30.0, "m2_dyn": 29.0, "m3": 2.0, "m1_ar": 1.12},
    ]
    lines = cr.tier1_table(rows, base_run=BD3_NAME).splitlines()
    assert _cells(lines[3]) == [
        NLD_NAME,
        "objective nld (B = 32, diffusion weight 0.3, AR-only 0.2)",
        "1.1500",
        "-4.17",
        "30.0",
        "29.0",
        "2.00",
        "-",
        "AR-mode M1 1.1200",
    ]


def test_tier1_ignores_m1_ar_of_a_run_that_is_not_nld() -> None:
    rows = [
        {"run": BD3_NAME, "m1": 1.2, "m2_static": 30.5, "m2_dyn": 29.0, "m3": 2.4, "m1_ar": 1.1},
        {"run": AR_NAME, "m1": 1.1, "m2_ar": 40.0, "m1_ar": 1.0},
    ]
    lines = cr.tier1_table(rows, base_run=BD3_NAME).splitlines()
    assert not any("AR-mode M1" in line for line in lines)


def test_tier1_change_of_an_nld_run_lists_only_the_differences_from_r00() -> None:
    name = "nld-q3_0.6B-B8-sh1-a0.5-ao0.1-lr1e-4-T2B"
    lines = cr.tier1_table([{"run": name, "m1": 1.3}], base_run=BD3_NAME).splitlines()
    cells = _cells(lines[2])
    assert cells[1] == "objective nld (token shift 1, diffusion weight 0.5, AR-only 0.1); lr 1e-4"
    assert cells[3] == "-"
    assert cells[8] == ""


def test_pick_lr_clear_winner_needs_no_note() -> None:
    lr, note = cr.pick_lr(_tier0_rows((1e-5, 1.30), (3e-5, 1.20), (1e-4, 1.25)))
    assert lr == 3e-5
    assert note == ""


def test_pick_lr_within_half_percent_takes_the_lower_lr() -> None:
    lr, note = cr.pick_lr(_tier0_rows((1e-5, 1.2050), (3e-5, 1.2000), (1e-4, 1.25)))
    assert lr == 1e-5
    assert note == cr.TIE_LR_NOTE


def test_pick_lr_flags_a_low_edge_that_beats_the_middle_by_two_percent() -> None:
    lr, note = cr.pick_lr(_tier0_rows((1e-5, 1.00), (3e-5, 1.05), (1e-4, 1.10)))
    assert lr == 1e-5
    assert note == cr.EDGE_NOTE


def test_pick_lr_flags_a_high_edge_that_beats_the_middle_by_two_percent() -> None:
    lr, note = cr.pick_lr(_tier0_rows((1e-5, 1.10), (3e-5, 1.05), (1e-4, 1.00)))
    assert lr == 1e-4
    assert note == cr.EDGE_NOTE


def test_pick_lr_edge_without_a_two_percent_gain_has_no_note() -> None:
    lr, note = cr.pick_lr(_tier0_rows((1e-5, 1.00), (3e-5, 1.01), (1e-4, 1.10)))
    assert lr == 1e-5
    assert note == ""


def test_pick_lr_ignores_missing_m1_and_rejects_empty_input() -> None:
    rows = [{"run": "a", "lr": 1e-5, "m1": None}, {"run": "b", "lr": 3e-5, "m1": 1.2}]
    assert cr.pick_lr(rows) == (3e-5, "")
    with pytest.raises(ValueError):
        cr.pick_lr([{"run": "a", "lr": 1e-5, "m1": None}])


def test_pick_lr_rejects_two_runs_with_the_same_lr() -> None:
    with pytest.raises(ValueError):
        cr.pick_lr(_tier0_rows((1e-5, 1.2), (1e-5, 1.3)))


def test_tier0_table_marks_the_pick_and_its_edge_note() -> None:
    rows = _tier0_rows((1e-5, 1.00), (3e-5, 1.05), (1e-4, 1.10))
    assert cr.tier0_table(rows).splitlines() == [
        HEADER_TIER0,
        SEP_TIER0,
        "| run-0 | 1e-5 | 1.0000 | picked; edge: consider one more step |",
        "| run-1 | 3e-5 | 1.0500 |  |",
        "| run-2 | 1e-4 | 1.1000 |  |",
    ]


def test_replace_table_keeps_text_after_the_table() -> None:
    text = "# T\n\n## Tier 0 (x)\n\n| old |\n|---|\n\nnote line\n\n## Tier 1\n"
    updated = cr.replace_table(text, "## Tier 0", "| new |\n|---|")
    assert updated == "# T\n\n## Tier 0 (x)\n\n| new |\n|---|\n\nnote line\n\n## Tier 1\n"


def test_replace_table_rejects_a_missing_heading_or_table() -> None:
    with pytest.raises(ValueError):
        cr.replace_table("# T\n", "## Tier 0", "| a |")
    with pytest.raises(ValueError):
        cr.replace_table("## Tier 0\n\ntext\n## Tier 1\n", "## Tier 0", "| a |")


def test_main_write_replaces_only_the_tier_tables(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    eval_dir = _write_eval_dir(tmp_path)
    logs = _write_logs(tmp_path)
    results_md = tmp_path / "RESULTS.md"
    results_md.write_text(RESULTS_MD, encoding="utf-8")

    status = cr.main(
        [
            "--eval-dir",
            str(eval_dir),
            "--logs-glob",
            str(logs / "*.out"),
            "--results-md",
            str(results_md),
            "--write",
        ]
    )

    assert status == 0
    new_text = results_md.read_text(encoding="utf-8")
    assert _lines_outside_tier_tables(new_text) == _lines_outside_tier_tables(RESULTS_MD)
    lines = new_text.splitlines()
    tier0_start = lines.index("## Tier 0 (0.6B, 0.5B tokens)")
    tier1_start = lines.index("## Tier 1 (0.6B, 2B tokens)")
    tier2_start = lines.index("## Tier 2")
    assert lines[tier0_start:tier1_start] == [
        "## Tier 0 (0.6B, 0.5B tokens)",
        HEADER_TIER0,
        SEP_TIER0,
        *EXPECTED_TIER0,
        "",
    ]
    assert lines[tier1_start:tier2_start] == [
        "## Tier 1 (0.6B, 2B tokens)",
        HEADER_TIER1,
        SEP_TIER1,
        *EXPECTED_TIER1,
        "",
    ]
    assert "qwen3-0.6B-base" in capsys.readouterr().err


def test_main_without_write_prints_tables_and_keeps_the_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    eval_dir = _write_eval_dir(tmp_path)
    logs = _write_logs(tmp_path)
    results_md = tmp_path / "RESULTS.md"
    results_md.write_text(RESULTS_MD, encoding="utf-8")

    status = cr.main(
        [
            "--eval-dir",
            str(eval_dir),
            "--logs-glob",
            str(logs / "*.out"),
            "--results-md",
            str(results_md),
        ]
    )

    assert status == 0
    assert results_md.read_text(encoding="utf-8") == RESULTS_MD
    out_lines = capsys.readouterr().out.splitlines()
    for line in EXPECTED_TIER0 + EXPECTED_TIER1:
        assert line in out_lines
    assert "pick: LR 3e-5" in out_lines
    assert f"base run: {BD3_NAME}" in out_lines


def test_main_adds_an_nld_run_to_tier_1_and_keeps_tier_0(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    eval_dir = _write_eval_dir(tmp_path)
    _write_json(eval_dir / f"{NLD_NAME}.m1.json", {"bits_per_token": 1.15})
    _write_json(eval_dir / f"{NLD_NAME}.m1_ar.json", {"bits_per_token": 1.12})
    logs = _write_logs(tmp_path)
    results_md = tmp_path / "RESULTS.md"
    results_md.write_text(RESULTS_MD, encoding="utf-8")

    status = cr.main(
        [
            "--eval-dir",
            str(eval_dir),
            "--logs-glob",
            str(logs / "*.out"),
            "--results-md",
            str(results_md),
        ]
    )

    assert status == 0
    out_lines = capsys.readouterr().out.splitlines()
    tier0_start = out_lines.index("## Tier 0")
    tier1_start = out_lines.index("## Tier 1")
    base_line = out_lines.index(f"base run: {BD3_NAME}")
    assert out_lines[tier0_start:tier1_start] == [
        "## Tier 0",
        HEADER_TIER0,
        SEP_TIER0,
        *EXPECTED_TIER0,
        "pick: LR 3e-5",
        "",
    ]
    assert out_lines[tier1_start:base_line] == [
        "## Tier 1",
        HEADER_TIER1,
        SEP_TIER1,
        *EXPECTED_TIER1,
        NLD_TIER1_ROW,
    ]


def test_main_rejects_a_missing_eval_directory(tmp_path: Path) -> None:
    results_md = tmp_path / "RESULTS.md"
    results_md.write_text(RESULTS_MD, encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        cr.main(
            [
                "--eval-dir",
                str(tmp_path / "absent"),
                "--results-md",
                str(results_md),
            ]
        )
