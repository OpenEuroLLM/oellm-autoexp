#!/usr/bin/env python3
"""Collect recall-suite eval numbers into ``dump/recall_evals.csv``.

Scans the MN5 eval tree for ``ambitious_<arm>_fp8_<step>`` directories, asks the
cluster aggregator for each one, adds the JUPITER 32B-dense reference CSVs and
the same-harness Qwen reference dirs, then rewrites the CSV (long format).

Re-run after new readouts land; the file is regenerated, never appended.

    python3 scripts/korbi/evals/collect_recall.py            # refresh CSV
    python3 scripts/korbi/evals/collect_recall.py --markdown # also print table

Columns: recorded_at, model, step, metric, value, rows_done, rows_expected, source
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

HOST = "marenostrum_new"
EVAL_ROOT = "/gpfs/scratch/ehpc533/hybrid_nucleus/flag_evals_oellm"
AGG = "/gpfs/projects/ehpc533/users/korbip/hybrid_v019/agg_evals.py"

JUPITER = "jupiter"
JUPITER_DIR = "/e/scratch/e-sta-openeurollm/poeppel1/flag-evals/results"
# 90k steps (~1.5T tokens) and 300k steps (~5T tokens) dense 32B checkpoints.
JUPITER_REFS = {
    "ref_olmo32b_90k": ("b1cons_90k.flag-evals-438.tasks.csv", 90000),
    "ref_olmo32b_300k": ("b1consTW_300k.flag-evals-438.tasks.csv", 300000),
}

# Same-harness references that were evaluated in-tree (no step number).
QWEN_REFS = {
    "qwen2.5-32b": "ext_Qwen2.5-32B",
    "qwen3-14b": "ext_Qwen3-14B-Base",
    "qwen3-30b-a3b": "ext_Qwen3-30B-A3B-Base",
    "qwen3.5-35b-a3b": "ext_Qwen3.5-35B-A3B-Base",
}

ARM_PREFIX = {
    "ambitious_v4_2ph": "v4",
    "ambitious_v4a_v6_pin": "v4a",
    "ambitious_v4a_v": "v4a",
    "ambitious_v4b_learned": "v4b",
    "ambitious_v4c_ramp": "v4c",
}

METRICS = [
    "coqa_f1",
    "coqa_em",
    "squadv2_f1",
    "squadv2_em",
    "squadv2_hasans_f1",
    "winogrande",
    "belebele_mean",
    "belebele_eng",
    "swde",
    "fda",
    "squad_completion",
]
ROWS_EXPECTED = 30

ARM_DIR_RE = re.compile(r"^(ambitious_v[0-9][a-z]?_[a-z0-9_]+?)_fp8_(\d+)$")


def ssh(host: str, script: str, timeout: int = 900) -> str:
    """Run a bash snippet on a remote host and return stdout."""
    out = subprocess.run(
        ["ssh", host, script],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if out.returncode != 0:
        print(f"warning: {host} exited {out.returncode}: {out.stderr[-300:]}", file=sys.stderr)
    return out.stdout


def collect_mn5() -> list[dict]:
    """Ask the cluster aggregator for every arm checkpoint and Qwen reference."""
    dirs = ssh(HOST, f"ls -d {EVAL_ROOT}/ambitious_v* {EVAL_ROOT}/ext_* 2>/dev/null")
    names = sorted({Path(p).name for p in dirs.split()})
    if not names:
        return []

    # One remote call for all directories: emit "DIR<TAB>rows" then the agg output.
    remote = [f"E={EVAL_ROOT}", f"A={AGG}"]
    for name in names:
        remote.append(f'echo "@@ {name}"')
        remote.append(f'echo "rows $(ls $E/{name}/done/*.ok 2>/dev/null | grep -vc "@")"')
        remote.append(f'python3 "$A" "$E/{name}" 2>/dev/null')
    raw = ssh(HOST, "\n".join(remote))

    records: list[dict] = []
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    current: dict | None = None
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("@@"):
            name = line[2:].strip()
            if name in QWEN_REFS.values():
                model, step = next(k for k, v in QWEN_REFS.items() if v == name), ""
            else:
                m = ARM_DIR_RE.match(name)
                if not m:
                    current = None
                    continue
                if m.group(1) not in ARM_PREFIX:
                    current = None
                    continue
                model, step = ARM_PREFIX[m.group(1)], int(m.group(2))
            current = {"model": model, "step": step, "rows_done": 0, "source": name}
            continue
        if current is None or not line:
            continue
        if line.startswith("rows "):
            # External reference trees hold many suites; their row count is not the
            # 30-row recall denominator, so it must not gate completeness.
            count = int(line.split()[1])
            current["rows_done"] = count if current["step"] != "" else ""
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0] in METRICS:
            try:
                value = float(parts[1])
            except ValueError:
                continue
            records.append(
                {
                    "recorded_at": stamp,
                    "model": current["model"],
                    "step": current["step"],
                    "metric": parts[0],
                    "value": value,
                    "rows_done": current["rows_done"],
                    "rows_expected": ROWS_EXPECTED if current["step"] != "" else "",
                    "source": current["source"],
                }
            )
    return records


JUPITER_PROGRAM = '''
import csv, sys
refs = {
    "ref_olmo32b_90k": ("%s", 90000),
    "ref_olmo32b_300k": ("%s", 300000),
}
metric_of = {
    ("coqa", "f1"): "coqa_f1",
    ("coqa", "em"): "coqa_em",
    ("squadv2", "f1"): "squadv2_f1",
    ("squadv2", "exact"): "squadv2_em",
    ("squadv2", "HasAns_f1"): "squadv2_hasans_f1",
    ("winogrande", "acc"): "winogrande",
    ("belebele_eng_Latn", "acc"): "belebele_eng",
}
for model, (fname, step) in refs.items():
    path = fname
    rows = list(csv.DictReader(open(path)))
    seen = set()
    for row in rows:
        key = metric_of.get((row["task"], row["metric"]))
        if key is None or key in seen:
            continue
        seen.add(key)
        value = float(row["value"])
        if key in ("coqa_f1", "coqa_em", "winogrande", "belebele_eng") and value <= 1.0:
            value *= 100.0
        print("%%s|%%s|%%s|%%.4f" %% (model, step, key, value))
    langs = [float(r["value"]) for r in rows if r["task"].startswith("belebele_") and r["metric"] == "acc"]
    if langs:
        print("%%s|%%s|belebele_mean|%%.4f" %% (model, step, sum(langs) / len(langs) * 100.0))
''' % (f"{JUPITER_DIR}/b1cons_90k.flag-evals-438.tasks.csv",
       f"{JUPITER_DIR}/b1consTW_300k.flag-evals-438.tasks.csv")


def collect_jupiter() -> list[dict]:
    """Read the 32B dense reference CSVs (coqa/squadv2/winogrande/belebele)."""
    out = subprocess.run(
        ["ssh", JUPITER, "python3 -"],
        input=JUPITER_PROGRAM,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if out.returncode != 0:
        print(f"warning: jupiter exited {out.returncode}: {out.stderr[-300:]}", file=sys.stderr)

    records: list[dict] = []
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    for line in out.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 4:
            continue
        model, step, metric, value = parts
        records.append(
            {
                "recorded_at": stamp,
                "model": model,
                "step": int(step),
                "metric": metric,
                "value": round(float(value), 4),
                "rows_done": "",
                "rows_expected": "",
                "source": JUPITER_REFS[model][0],
            }
        )
    return records


def write_csv(records: list[dict], path: Path) -> None:
    fields = [
        "recorded_at",
        "model",
        "step",
        "metric",
        "value",
        "rows_done",
        "rows_expected",
        "source",
    ]
    records.sort(key=lambda r: (str(r["model"]), str(r["step"]).zfill(6), r["metric"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def print_markdown(records: list[dict]) -> None:
    """Print one markdown table: metrics x (model, step) for complete rows only."""
    complete = [r for r in records if r["rows_done"] in ("", ROWS_EXPECTED)]
    columns: list[tuple[str, str]] = []
    for r in complete:
        key = (str(r["model"]), str(r["step"]))
        if key not in columns:
            columns.append(key)
    columns.sort(key=lambda c: (c[0], c[1].zfill(6)))
    columns = [(m, s) for m, s in columns if s != ""] + [(m, s) for m, s in columns if s == ""]

    lookup = {(str(r["model"]), str(r["step"]), r["metric"]): r["value"] for r in complete}
    header = "| metric | " + " | ".join(
        f"{model} @{step}" if step else model for model, step in columns
    ) + " |"
    print(header)
    print("|" + "---|" * (len(columns) + 1))
    for metric in METRICS:
        cells = []
        for model, step in columns:
            value = lookup.get((model, step, metric))
            cells.append("–" if value is None else f"{value:.2f}")
        print(f"| {metric} | " + " | ".join(cells) + " |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="dump/recall_evals.csv", help="output CSV path")
    parser.add_argument("--markdown", action="store_true", help="print a markdown table")
    args = parser.parse_args()

    records = collect_mn5() + collect_jupiter()
    if not records:
        print("no records collected", file=sys.stderr)
        return 1
    out = Path(args.out)
    write_csv(records, out)
    models = sorted({r["model"] for r in records})
    print(f"wrote {len(records)} rows for {len(models)} models to {out}")
    if args.markdown:
        print()
        print_markdown(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
