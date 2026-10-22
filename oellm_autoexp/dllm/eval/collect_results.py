"""Collect evaluation JSON files and training logs into the RESULTS.md tables.

Metrics per run (see ``.claude/dllm/ABLATIONS.md``):

* M1 is the field ``bits_per_token`` of ``<run>.m1.json``.
* M2 static and M2 dyn@0.9 are the field ``accuracy`` (percent) of
  ``<run>.gsm8k_static.json`` and ``<run>.gsm8k_dynamic.json``.
* M3 is the field ``mean_tpf_decode`` of ``<run>.gsm8k_dynamic.json``.
* M4 is the median ``tokens/s/GPU`` of the ``bd throughput`` lines of the training log.

Run from the repository root::

    python -m oellm_autoexp.dllm.eval.collect_results --eval-dir DIR --results-md RESULTS.md

Without ``--write`` the tables are only printed. With ``--write`` the table under ``## Tier 0``
and the table under ``## Tier 1`` are replaced, and every other line is kept. This module
needs no torch.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

EDGE_NOTE = "edge: consider one more step"
TIE_LR_NOTE = "tie within 0.5 percent: lower LR picked"
TIE_LR_PERCENT = 0.5
EDGE_BEAT_PERCENT = 2.0
TIE_DM1_PERCENT = 0.3
TIER_SIZE = "0.6B"
TIER0_TOKENS = 500_000_000
TIER1_TOKENS = 2_000_000_000

# Base configuration R00 (ABLATIONS.md). Tier 1 changes are described relative to it.
BASE_BD3: dict[str, Any] = {
    "block_size": 8,
    "token_shift": 0,
    "complementary": 0,
    "weighting": "elbo",
    "t_min": 0.0,
    "t_max": 1.0,
    "doc_mask": 1,
    "schedule": None,
}

TIER0_HEADER = ("| Run | LR | M1 | Notes |", "|---|---|---|---|")
TIER1_HEADER = (
    "| Run | Change | M1 | dM1 vs R00 (%) | M2 static | M2 dyn@0.9 | M3 | M4 | Notes |",
    "|---|---|---|---|---|---|---|---|---|",
)

_TOKEN_SCALE = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
_NUM = r"\d+(?:\.\d+)?"
_SIZE = rf"{_NUM}B(?:_[a-z]+)?"
_LR = rf"{_NUM}(?:e[-+]?\d+)?"
_TOKENS = rf"{_NUM}[KMB]"
# Run names may spell the schedule 0:4,0.15:8 as 0-4_0.15-8, which keeps ':' and ',' out of paths.
_SCHEDULE = rf"{_NUM}:\d+(?:,{_NUM}:\d+)*|{_NUM}-\d+(?:_{_NUM}-\d+)*"
_BD3_RE = re.compile(
    rf"bd3-q3_(?P<size>{_SIZE})-B(?P<block_size>\d+)"
    rf"-sh(?P<token_shift>[01])-cm(?P<complementary>[01])"
    rf"-w(?P<weighting>elbo|uniform)-t(?P<t_min>{_NUM})_(?P<t_max>{_NUM})"
    rf"-dm(?P<doc_mask>[01])-sc(?P<schedule>none|{_SCHEDULE})"
    rf"-lr(?P<lr>{_LR})-T(?P<tokens>{_TOKENS})"
)
_AR_RE = re.compile(rf"ar-q3_(?P<size>{_SIZE})-lr(?P<lr>{_LR})-T(?P<tokens>{_TOKENS})")
_THROUGHPUT_RE = re.compile(
    r"bd throughput \| iteration\s+\d+ \| tokens/s/GPU: (?P<value>[-+]?\d+(?:\.\d+)?)"
)


def _schedule_text(text: str) -> str | None:
    """Return the schedule field of a run name as ``--bd-block-schedule`` text,
    or None for ``none``.

    The path-safe spelling ``0-4_0.15-8`` becomes ``0:4,0.15:8``.
    """
    if text == "none":
        return None
    if ":" in text:
        return text
    return ",".join(item.replace("-", ":") for item in text.split("_"))


def _parse_lr(text: str, name: str) -> float:
    """Convert the LR field of a run name to a positive float."""
    lr = float(text)
    if not lr > 0.0:
        raise ValueError(f"learning rate must be positive in run name {name!r}")
    return lr


def _parse_tokens(text: str, name: str) -> int:
    """Convert the token field of a run name ("0.5B", "2B", "500M") to a token
    count."""
    count = round(float(text[:-1]) * _TOKEN_SCALE[text[-1]])
    if count < 1:
        raise ValueError(f"token count must be positive in run name {name!r}")
    return count


def parse_run_name(name: str) -> dict[str, Any]:
    """Parse a block-diffusion or AR run name from ``ABLATIONS.md``.

    The accepted patterns are
    ``bd3-q3_<size>-B<B>-sh<shift>-cm<comp>-w<weight>-t<tmin>_<tmax>-dm<doc>-sc<schedule|none>``
    ``-lr<lr>-T<tokens>`` and ``ar-q3_<size>-lr<lr>-T<tokens>``.

    Args:
        name: The run name.

    Returns:
        A dict with the keys ``objective`` ("bd3" or "ar"), ``size`` (for example "0.6B"),
        ``block_size`` (int), ``token_shift`` (0 or 1), ``complementary`` (0 or 1),
        ``weighting`` ("elbo" or "uniform"), ``t_min`` and ``t_max`` (float), ``doc_mask``
        (0 or 1), ``schedule`` (the schedule string, or None for a constant block size),
        ``lr`` (float), and ``tokens`` (int, for example "2B" gives 2000000000). For AR names
        the block-diffusion fields are None.

    Raises:
        ValueError: If the name matches neither pattern, or a value is out of range.
    """
    match = _BD3_RE.fullmatch(name)
    if match is not None:
        groups = match.groupdict()
        block_size = int(groups["block_size"])
        t_min = float(groups["t_min"])
        t_max = float(groups["t_max"])
        if block_size < 1 or not 0.0 <= t_min <= t_max <= 1.0:
            raise ValueError(f"block size or t range out of range in run name {name!r}")
        return {
            "objective": "bd3",
            "size": groups["size"],
            "block_size": block_size,
            "token_shift": int(groups["token_shift"]),
            "complementary": int(groups["complementary"]),
            "weighting": groups["weighting"],
            "t_min": t_min,
            "t_max": t_max,
            "doc_mask": int(groups["doc_mask"]),
            "schedule": _schedule_text(groups["schedule"]),
            "lr": _parse_lr(groups["lr"], name),
            "tokens": _parse_tokens(groups["tokens"], name),
        }
    match = _AR_RE.fullmatch(name)
    if match is not None:
        groups = match.groupdict()
        return {
            "objective": "ar",
            "size": groups["size"],
            "block_size": None,
            "token_shift": None,
            "complementary": None,
            "weighting": None,
            "t_min": None,
            "t_max": None,
            "doc_mask": None,
            "schedule": None,
            "lr": _parse_lr(groups["lr"], name),
            "tokens": _parse_tokens(groups["tokens"], name),
        }
    raise ValueError(f"not a block-diffusion or AR run name: {name!r}")


def format_lr(lr: float) -> str:
    """Format a learning rate compactly, for example 1e-5 or 3e-4.

    Args:
        lr: The learning rate.

    Returns:
        The mantissa and the exponent without zero padding, joined by "e".
    """
    mantissa, _, exponent = f"{lr:e}".partition("e")
    return f"{float(mantissa):g}e{int(exponent)}"


def _throughput_values(log_text: str) -> list[float]:
    """Return the tokens/s/GPU value of every ``bd throughput`` line, in log
    order."""
    return [float(match.group("value")) for match in _THROUGHPUT_RE.finditer(log_text)]


def _median_after_skip(series: Iterable[Sequence[float]], skip_first: int) -> float | None:
    """Return the median of the values left after dropping the first values of
    each series."""
    kept = [value for values in series for value in values[skip_first:]]
    if not kept:
        return None
    return float(statistics.median(kept))


def tokens_per_s_gpu(log_text: str, skip_first: int = 2) -> float | None:
    """Return the median training tokens/s/GPU of one log (metric M4).

    Args:
        log_text: Text of a training log with ``bd throughput`` lines.
        skip_first: Number of leading throughput values to drop as warm-up.

    Returns:
        The median of the remaining values, or None when no value remains.

    Raises:
        ValueError: If ``skip_first`` is negative.
    """
    if skip_first < 0:
        raise ValueError(f"skip_first must be >= 0, got {skip_first}")
    return _median_after_skip([_throughput_values(log_text)], skip_first)


def _read_number(path: Path, key: str) -> float:
    """Read one numeric field of a JSON object file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or key not in data:
        raise ValueError(f"{path}: missing field {key!r}")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{path}: field {key!r} is not a number: {value!r}")
    return float(value)


def _entry(results: dict[str, dict[str, float | None]], run: str) -> dict[str, float | None]:
    """Return the metric entry of a run, creating it with empty fields when
    needed."""
    return results.setdefault(
        run, {"m1": None, "m2_static": None, "m2_dyn": None, "m3": None, "m2_ar": None}
    )


def load_eval_dir(eval_dir: str | os.PathLike[str]) -> dict[str, dict[str, float | None]]:
    """Read the evaluation JSON files of every run in a directory.

    The files are matched by suffix. ``<run>.m1.json`` gives ``m1`` (field
    ``bits_per_token``). ``<run>.gsm8k_static.json`` gives ``m2_static`` (field ``accuracy``
    in percent). ``<run>.gsm8k_dynamic.json`` gives ``m2_dyn`` (``accuracy``) and ``m3``
    (``mean_tpf_decode``). ``<run>.gsm8k_ar.json`` (AR greedy decoding, the M2 of AR runs) gives
    ``m2_ar`` (``accuracy``). Other files are ignored.

    Args:
        eval_dir: Directory with the evaluation JSON files.

    Returns:
        Maps each run name to ``{"m1", "m2_static", "m2_dyn", "m3", "m2_ar"}``. A field whose
        file is absent is None.

    Raises:
        FileNotFoundError: If ``eval_dir`` is not a directory.
        ValueError: If a matched file lacks its field.
        TypeError: If a matched field is not a number.
    """
    root = Path(eval_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"evaluation directory not found: {root}")
    results: dict[str, dict[str, float | None]] = {}
    for path in sorted(root.glob("*.json")):
        if not path.is_file():
            continue
        name = path.name
        if name.endswith(".m1.json"):
            entry = _entry(results, name[: -len(".m1.json")])
            entry["m1"] = _read_number(path, "bits_per_token")
        elif name.endswith(".gsm8k_static.json"):
            entry = _entry(results, name[: -len(".gsm8k_static.json")])
            entry["m2_static"] = _read_number(path, "accuracy")
        elif name.endswith(".gsm8k_dynamic.json"):
            entry = _entry(results, name[: -len(".gsm8k_dynamic.json")])
            entry["m2_dyn"] = _read_number(path, "accuracy")
            entry["m3"] = _read_number(path, "mean_tpf_decode")
        elif name.endswith(".gsm8k_ar.json"):
            entry = _entry(results, name[: -len(".gsm8k_ar.json")])
            entry["m2_ar"] = _read_number(path, "accuracy")
    return results


def collect_throughput(
    logs_glob: str,
    runs: Iterable[str],
    skip_first: int = 2,
) -> dict[str, float | None]:
    """Map each run to the median tokens/s/GPU over its training log(s) (metric
    M4).

    A log belongs to the longest run name that occurs in its file name. The first
    ``skip_first`` throughput values of each log are dropped as warm-up, and the medians are
    taken over the values of all logs of the run.

    Args:
        logs_glob: Glob pattern of training logs. Environment variables are expanded. An
            empty pattern matches no logs.
        runs: Run names to look up.
        skip_first: Number of leading throughput values to drop from each log.

    Returns:
        Maps each run to its median tokens/s/GPU, or None when no log matched it.

    Raises:
        ValueError: If ``skip_first`` is negative.
    """
    if skip_first < 0:
        raise ValueError(f"skip_first must be >= 0, got {skip_first}")
    run_list = list(runs)
    series: dict[str, list[list[float]]] = {run: [] for run in run_list}
    if logs_glob:
        pattern = os.path.expanduser(os.path.expandvars(logs_glob))
        for name in sorted(glob.glob(pattern, recursive=True)):
            path = Path(name)
            if not path.is_file():
                continue
            candidates = [run for run in run_list if run in path.name]
            if not candidates:
                continue
            log_text = path.read_text(encoding="utf-8", errors="replace")
            series[max(candidates, key=len)].append(_throughput_values(log_text))
    return {run: _median_after_skip(series[run], skip_first) for run in run_list}


def _fmt(value: float | None, spec: str) -> str:
    """Format a metric value, or return "-" when it is missing."""
    return "-" if value is None else format(value, spec)


def _is_base_bd3(row: Mapping[str, Any]) -> bool:
    """Return True if a parsed row is a bd3 run with the R00 configuration (any
    LR)."""
    if row["objective"] != "bd3":
        return False
    return all(row[key] == value for key, value in BASE_BD3.items())


def _describe_change(parsed: Mapping[str, Any], base_lr: float | None) -> str:
    """Describe how a parsed run differs from the R00 configuration.

    Args:
        parsed: The output of :func:`parse_run_name` for the run.
        base_lr: LR of the base run, or None to leave the LR out of the description.

    Returns:
        The differences joined by "; ", or "none" when there are none.
    """
    parts: list[str] = []
    if parsed["objective"] != "bd3":
        parts.append(f"objective {parsed['objective']}")
    else:
        if parsed["block_size"] != BASE_BD3["block_size"]:
            parts.append(f"B = {parsed['block_size']}")
        if parsed["token_shift"] != BASE_BD3["token_shift"]:
            parts.append(f"token shift {parsed['token_shift']}")
        if parsed["complementary"] != BASE_BD3["complementary"]:
            parts.append(f"complementary mask {parsed['complementary']}")
        if parsed["weighting"] != BASE_BD3["weighting"]:
            parts.append(f"{parsed['weighting']} weighting")
        if (parsed["t_min"], parsed["t_max"]) != (BASE_BD3["t_min"], BASE_BD3["t_max"]):
            parts.append(f"t in [{parsed['t_min']:g}, {parsed['t_max']:g}]")
        if parsed["doc_mask"] != BASE_BD3["doc_mask"]:
            parts.append(f"doc mask {parsed['doc_mask']}")
        if parsed["schedule"] != BASE_BD3["schedule"]:
            parts.append(f"schedule {parsed['schedule']}")
    if base_lr is not None and not math.isclose(parsed["lr"], base_lr, rel_tol=1e-9):
        parts.append(f"lr {format_lr(parsed['lr'])}")
    return "; ".join(parts) if parts else "none"


def pick_lr(rows: Sequence[Mapping[str, Any]]) -> tuple[float, str]:
    """Pick the tier 0 learning rate by the rule in ``ABLATIONS.md``.

    The best run has the lowest M1. Every run whose M1 is within 0.5 percent of that value
    ties with it, and the tie goes to the lower LR. If the picked LR is the lowest or the
    highest of the sorted LR list (the edge rule needs at least three runs) and its M1 beats
    the M1 of the middle LR by more than 2 percent, the note asks for one more step.

    Args:
        rows: Tier 0 rows with ``lr`` (float) and ``m1`` (float or None). Rows without M1
            are ignored.

    Returns:
        The picked LR and a note. The note is empty, or the tie and edge notes joined by "; ".

    Raises:
        ValueError: If no row has an M1 value, or two rows with an M1 value share an LR.
    """
    measured = [row for row in rows if row.get("m1") is not None]
    if not measured:
        raise ValueError("pick_lr needs at least one run with an M1 value")
    lrs = [row["lr"] for row in measured]
    if len(set(lrs)) != len(lrs):
        raise ValueError(f"pick_lr: two runs share an LR: {sorted(lrs)}")
    best = min(row["m1"] for row in measured)
    tied = [row for row in measured if row["m1"] <= best * (1 + TIE_LR_PERCENT / 100)]
    picked = min(tied, key=lambda row: row["lr"])
    ordered = sorted(measured, key=lambda row: row["lr"])
    notes: list[str] = []
    if len(tied) > 1:
        notes.append(TIE_LR_NOTE)
    if len(ordered) >= 3 and picked["lr"] in (ordered[0]["lr"], ordered[-1]["lr"]):
        middle = ordered[len(ordered) // 2]
        if middle["m1"] - picked["m1"] > middle["m1"] * EDGE_BEAT_PERCENT / 100:
            notes.append(EDGE_NOTE)
    return float(picked["lr"]), "; ".join(notes)


def tier0_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render the Tier 0 table (Run | LR | M1 | Notes), sorted by LR.

    The run that :func:`pick_lr` picks gets the note "picked", followed by the pick note.

    Args:
        rows: Tier 0 rows with ``run``, ``lr``, and ``m1`` (None when missing).

    Returns:
        The markdown table, without a trailing newline.
    """
    picked_lr: float | None = None
    pick_note = ""
    if any(row.get("m1") is not None for row in rows):
        picked_lr, pick_note = pick_lr(rows)
    lines = list(TIER0_HEADER)
    for row in sorted(rows, key=lambda item: item["lr"]):
        m1_text = _fmt(row.get("m1"), ".4f")
        notes = ""
        if picked_lr is not None and row.get("m1") is not None and row["lr"] == picked_lr:
            notes = "; ".join(part for part in ("picked", pick_note) if part)
        lines.append(f"| {row['run']} | {format_lr(row['lr'])} | {m1_text} | {notes} |")
    return "\n".join(lines)


def tier1_table(rows: Sequence[Mapping[str, Any]], base_run: str | None) -> str:
    """Render the Tier 1 table (Run | Change | M1 | dM1 vs R00 (%) | M2 ... |
    Notes).

    dM1 = 100 * (M1 - M1_base) / M1_base, where the base is the row named ``base_run``. The
    base row is marked "base". Another row whose |dM1| is below 0.3 percent is marked "tie".
    The Change column lists the differences from the R00 configuration, and the LR difference
    from the base run. An AR run shows its AR greedy GSM8K accuracy (``m2_ar``) in the M2 static
    column, marked "M2 = AR greedy".

    Args:
        rows: Tier 1 rows with ``run``, ``m1``, ``m2_static``, ``m2_dyn``, ``m3``, ``m4``, and
            ``m2_ar`` (each may be None or absent).
        base_run: Name of the R00 run, or None when there is no base yet. Without a base,
            dM1 shows "-".

    Returns:
        The markdown table, without a trailing newline.

    Raises:
        ValueError: If the base run has M1 equal to zero.
    """
    base_row = None
    if base_run is not None:
        base_row = next((row for row in rows if row["run"] == base_run), None)
    base_m1 = None if base_row is None else base_row.get("m1")
    base_lr = None if base_run is None else parse_run_name(base_run)["lr"]
    lines = list(TIER1_HEADER)
    for row in rows:
        is_base = base_run is not None and row["run"] == base_run
        m1 = row.get("m1")
        dm1: float | None = None
        notes: list[str] = ["base"] if is_base else []
        if m1 is not None and base_m1 is not None:
            if base_m1 == 0:
                raise ValueError(f"base run {base_run!r} has M1 = 0, so dM1 is undefined")
            dm1 = 100.0 * (m1 - base_m1) / base_m1
            if not is_base and abs(dm1) < TIE_DM1_PERCENT:
                notes.append("tie")
        parsed = parse_run_name(row["run"])
        m2_static = row.get("m2_static")
        if parsed["objective"] == "ar":
            m2_static = row.get("m2_ar")
            notes.append("M2 = AR greedy")
        cells = [
            str(row["run"]),
            _describe_change(parsed, base_lr),
            _fmt(m1, ".4f"),
            _fmt(dm1, "+.2f"),
            _fmt(m2_static, ".1f"),
            _fmt(row.get("m2_dyn"), ".1f"),
            _fmt(row.get("m3"), ".2f"),
            _fmt(row.get("m4"), ".0f"),
            "; ".join(notes),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def replace_table(markdown: str, heading: str, table: str) -> str:
    """Replace the markdown table under one heading and keep every other line.

    The table is the first run of lines that start with "|" after the heading and before the
    next line that starts with "## ".

    Args:
        markdown: Full text of the results file.
        heading: Start of the heading line, for example ``"## Tier 0"``. The heading must be
            followed by a space or the end of the line.
        table: Replacement table, one row per line, without a trailing newline.

    Returns:
        The updated text. Lines outside the replaced table keep their text and line endings.

    Raises:
        ValueError: If the heading does not occur exactly once, or no table follows it.
    """
    lines = markdown.splitlines(keepends=True)
    pattern = re.compile(re.escape(heading) + r"(?=\s|$)")
    found = [index for index, line in enumerate(lines) if pattern.match(line.rstrip("\r\n"))]
    if len(found) != 1:
        raise ValueError(f"expected one line starting with {heading!r}, found {len(found)}")
    start = found[0] + 1
    end = next(
        (index for index in range(start, len(lines)) if lines[index].startswith("## ")),
        len(lines),
    )
    first = next(
        (index for index in range(start, end) if lines[index].lstrip().startswith("|")),
        None,
    )
    if first is None:
        raise ValueError(f"no markdown table under {heading!r}")
    last = first
    while last < end and lines[last].lstrip().startswith("|"):
        last += 1
    new_lines = [row + "\n" for row in table.split("\n")]
    return "".join(lines[:first] + new_lines + lines[last:])


def main(argv: Sequence[str] | None = None) -> int:
    """Build the Tier 0 and Tier 1 tables, print them, and optionally write
    them.

    Tier 0 holds the bd3 runs with the R00 configuration at size 0.6B and 0.5B tokens. Tier 1
    holds every run at size 0.6B and 2B tokens. Its base run is the R00 configuration at the LR
    that :func:`pick_lr` picks from Tier 0. Evaluation files whose names are not block-diffusion
    or AR run names (for example reference models) are reported on stderr and skipped.

    Args:
        argv: Command-line arguments. None means ``sys.argv[1:]``.

    Returns:
        The exit status, 0 on success.

    Raises:
        ValueError: If more than one Tier 1 base run has the picked LR.
    """
    parser = argparse.ArgumentParser(description="Build the RESULTS.md tier tables.")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="directory with <run>.m1.json and <run>.gsm8k_*.json files",
    )
    parser.add_argument(
        "--logs-glob",
        default="",
        help="glob of training logs (quote it); each log name must contain its run name",
    )
    parser.add_argument(
        "--results-md",
        type=Path,
        required=True,
        help="RESULTS.md to read, and to update with --write",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="replace the tables under '## Tier 0' and '## Tier 1' in --results-md",
    )
    args = parser.parse_args(argv)

    eval_results = load_eval_dir(args.eval_dir)
    runs: list[str] = []
    for run in sorted(eval_results):
        try:
            parse_run_name(run)
        except ValueError:
            print(f"skipped {run!r}: not a block-diffusion or AR run name", file=sys.stderr)
            continue
        runs.append(run)
    throughput = collect_throughput(args.logs_glob, runs)
    rows = [
        {**parse_run_name(run), "run": run, **eval_results[run], "m4": throughput[run]}
        for run in runs
    ]
    tier0 = [
        row
        for row in rows
        if row["size"] == TIER_SIZE and row["tokens"] == TIER0_TOKENS and _is_base_bd3(row)
    ]
    tier1 = [row for row in rows if row["size"] == TIER_SIZE and row["tokens"] == TIER1_TOKENS]

    picked_lr: float | None = None
    pick_note = ""
    base_run: str | None = None
    measured0 = [row for row in tier0 if row.get("m1") is not None]
    if measured0:
        picked_lr, pick_note = pick_lr(measured0)
        matches = [
            row["run"]
            for row in tier1
            if _is_base_bd3(row) and math.isclose(row["lr"], picked_lr, rel_tol=1e-9)
        ]
        if len(matches) > 1:
            raise ValueError(
                f"more than one Tier 1 base run at LR {format_lr(picked_lr)}: {matches}"
            )
        base_run = matches[0] if matches else None

    tier0_text = tier0_table(tier0)
    tier1_sorted = sorted(tier1, key=lambda row: (row["run"] != base_run, row["run"]))
    tier1_text = tier1_table(tier1_sorted, base_run)

    print("## Tier 0")
    print(tier0_text)
    if picked_lr is None:
        print("pick: no Tier 0 run has an M1 value yet")
    else:
        note = f" ({pick_note})" if pick_note else ""
        print(f"pick: LR {format_lr(picked_lr)}{note}")
    print()
    print("## Tier 1")
    print(tier1_text)
    print(f"base run: {base_run if base_run is not None else '-'}")

    if args.write:
        text = args.results_md.read_text(encoding="utf-8")
        text = replace_table(text, "## Tier 0", tier0_text)
        text = replace_table(text, "## Tier 1", tier1_text)
        args.results_md.write_text(text, encoding="utf-8")
        print(f"updated the Tier 0 and Tier 1 tables in {args.results_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
