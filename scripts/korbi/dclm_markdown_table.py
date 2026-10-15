#!/usr/bin/env python3
"""Markdown table of the dclm-core-22 results from all_mlscaling_results.csv.

Reads the combined CSV written by ``collect_all_results.py`` and emits a
markdown report restricted to:

  * ``eval_type == evalcd`` -- the cooldown endpoints, i.e. the only rows that
    have downstream numbers at all (stable-branch rows exist in the CSV but
    carry no task scores by construction).
  * the **dclm-core-22** task set, taken from
    ``submodules/oellm_evals/oellm/resources/task-groups.yaml`` rather than
    inferred from which columns happen to be populated -- so a task that failed
    everywhere shows up as an empty column instead of silently vanishing.

Rows are sorted by (model size, data size, variant), so every variant for a
given size/budget cell sits together and is read down the column.

Parameter counts come from ``scripts/korbi/scaling_laws/costs.py`` (the shared
per-size Qwen3 geometry). One arm is not param-matched: gdn3_qwen35's attention
output gate adds ``hidden * heads * kv_channels`` per attention layer, which is
added explicitly here instead of being quietly ignored.

Usage
-----
    python scripts/korbi/dclm_markdown_table.py \\
        --csv dump/all_mlscaling_results.csv \\
        --out dump/dclm_results.md
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import pathlib
import statistics
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dclm_task_columns() -> list[str]:
    """The dclm-core-22 task@Nshot keys, from the authoritative group yaml."""
    import yaml

    y = _REPO / "submodules/oellm_evals/oellm/resources/task-groups.yaml"
    doc = yaml.safe_load(y.read_text())
    grp = None
    for k, v in doc.items():
        if k == "dclm-core-22":
            grp = v
        elif isinstance(v, dict) and "dclm-core-22" in v:
            grp = v["dclm-core-22"]
    if grp is None:
        raise SystemExit("dclm-core-22 not found in task-groups.yaml")
    out = []
    for t in grp["tasks"]:
        for s in t.get("n_shots", [0]):
            out.append(f"{t['task']}@{s}shot")
    return sorted(out)


def param_counts(costs, variant: str, size: str) -> tuple[int, int]:
    """(non-embedding, total) params.

    Adds the qwen35 attention-gate delta.
    """
    n_ne = costs.n_ne(size)
    n_tot = costs.n_total(size)
    if variant == "gdn3_qwen35":
        a = costs.SIZE_ARCH[size]
        freq = 4  # 3 linear : 1 attention
        n_attn = a["L"] // freq
        gate = a["h"] * a["heads"] * costs.KV_CHANNELS * n_attn
        n_ne += gate
        n_tot += gate
    return n_ne, n_tot


def rank_marks(values: dict[int, float], higher_is_better: bool) -> dict[int, str]:
    """Map row-id -> "best" / "second" within one (size, data) group.

    Ranks by DISTINCT value, so a tie for first marks every tied row as
    best and the next distinct value becomes second -- rather than
    silently promoting one of the tied rows and hiding the tie.
    """
    if not values:
        return {}
    distinct = sorted({v for v in values.values()}, reverse=higher_is_better)
    marks: dict[int, str] = {}
    for rid, v in values.items():
        if v == distinct[0]:
            marks[rid] = "best"
        elif len(distinct) > 1 and v == distinct[1]:
            marks[rid] = "second"
    return marks


def decorate(text: str, mark: str | None) -> str:
    if mark == "best":
        return f"**{text}**"
    if mark == "second":
        return f"<u>{text}</u>"
    return text


def _fmt(v, nd=4):
    if v in (None, ""):
        return "--"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "--"


def _m(n: int) -> str:
    return f"{n / 1e6:.0f}M"


def _size_key(s: str) -> float:
    try:
        return float(s.rstrip("B"))
    except ValueError:
        return 0.0


def size_rule(ncols: int) -> str:
    """A dashed row used to separate model-size blocks inside a table.

    Markdown has no way to put a real <hr> inside a table, and an empty
    row is dropped by several renderers -- a row of em-dashes survives
    everywhere and reads as a rule.
    """
    return "| " + " | ".join(["———"] * ncols) + " |"


def _groups(rows: list[dict]) -> list[list[int]]:
    """Row indices grouped by (size, data size) -- the comparison unit."""
    out: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        out.setdefault((r["size"], r["budget_bt"]), []).append(i)
    return list(out.values())


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--csv", type=pathlib.Path, default=pathlib.Path("dump/all_mlscaling_results.csv")
    )
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("dump/dclm_results.md"))
    p.add_argument(
        "--short-names",
        action="store_true",
        help="abbreviate task column headers in the wide table",
    )
    args = p.parse_args()

    if not args.csv.exists():
        print(f"ERROR: {args.csv} not found", file=sys.stderr)
        return 1
    costs = _load(_REPO / "scripts/korbi/scaling_laws/costs.py", "sl_costs")
    tasks = dclm_task_columns()

    rows = [r for r in csv.DictReader(args.csv.open()) if r.get("eval_type") == "evalcd"]
    present = [t for t in tasks if t in (rows[0].keys() if rows else {})]
    missing_cols = [t for t in tasks if t not in present]

    # sort by model size, then data size, then variant
    rows.sort(key=lambda r: (_size_key(r["size"]), int(r["budget_bt"]), r["variant"]))

    out: list[str] = ["# DCLM-core results (cooldown endpoints)\n"]
    out.append(
        f"{len(rows)} runs, `eval_type=evalcd` only. Task set: **dclm-core-22** "
        f"({len(tasks)} task x n-shot combinations, {len(present)} present in the CSV).\n"
    )
    if missing_cols:
        out.append(
            "Never produced by any run (lm-eval issues, identical for every arm, so "
            "comparisons stay matched): " + ", ".join(f"`{c}`" for c in missing_cols) + "\n"
        )
    out.append(
        "\n`N_ne` is the non-embedding parameter count from the shared per-size "
        "geometry; `gdn3_qwen35` includes its attention-gate surplus "
        "(`hidden*heads*kv_channels` per attention layer), which is real and makes "
        "that arm 2-3% larger than the others at the same nominal size.\n"
    )

    # ---------------- summary ----------------
    out.append("\n## Summary\n")
    out.append(
        "\nBest and second best within each (size, data) group: "
        "**bold** = best, <u>underlined</u> = second. val_loss ranks LOW-is-better, "
        "accuracies HIGH-is-better.\n"
    )
    out.append("| size | data (BT) | variant | N_ne | N_total | val_loss | dclm_mean | n_tasks |")
    out.append("|---|---:|---|---:|---:|---:|---:|---:|")
    means = {}
    for i, r in enumerate(rows):
        vals = [float(r[t]) for t in present if r.get(t) not in (None, "")]
        means[i] = statistics.mean(vals) if vals else None
    prev_size = None
    for grp in _groups(rows):
        cur_size = rows[grp[0]]["size"]
        if prev_size is not None and cur_size != prev_size:
            out.append(size_rule(8))
        prev_size = cur_size
        vl = {i: float(rows[i]["val_loss"]) for i in grp if rows[i].get("val_loss")}
        dm = {i: means[i] for i in grp if means[i] is not None}
        mk_vl = rank_marks(vl, higher_is_better=False)
        mk_dm = rank_marks(dm, higher_is_better=True)
        for i in grp:
            r = rows[i]
            n_ne, n_tot = param_counts(costs, r["variant"], r["size"])
            n = sum(1 for t in present if r.get(t) not in (None, ""))
            out.append(
                f"| {r['size']} | {r['budget_bt']} | `{r['variant']}` | {_m(n_ne)} | {_m(n_tot)} | "
                f"{decorate(_fmt(r.get('val_loss')), mk_vl.get(i))} | "
                f"{decorate(_fmt(means[i]), mk_dm.get(i))} | {n} |"
            )

    # ---------------- per-task ----------------
    out.append("\n## Per-task\n")
    hdr = [
        t.replace("@", " @").replace("bigbench_", "bb_").replace("_generate_until", "*")
        if args.short_names
        else t
        for t in present
    ]
    out.append("| size | data (BT) | variant | " + " | ".join(hdr) + " | **mean** |")
    out.append("|---|---:|---|" + "---:|" * (len(present) + 1))
    prev_size = None
    for grp in _groups(rows):
        cur_size = rows[grp[0]]["size"]
        if prev_size is not None and cur_size != prev_size:
            out.append(size_rule(len(present) + 4))
        prev_size = cur_size
        per_task_marks = {}
        for t in present:
            col = {i: float(rows[i][t]) for i in grp if rows[i].get(t) not in (None, "")}
            per_task_marks[t] = rank_marks(col, higher_is_better=True)
        mk_mean = rank_marks(
            {i: means[i] for i in grp if means[i] is not None}, higher_is_better=True
        )
        for i in grp:
            r = rows[i]
            cells = [decorate(_fmt(r.get(t), 4), per_task_marks[t].get(i)) for t in present]
            out.append(
                f"| {r['size']} | {r['budget_bt']} | `{r['variant']}` | "
                + " | ".join(cells)
                + f" | {decorate(_fmt(means[i]), mk_mean.get(i))} |"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n")
    print(f"rows (evalcd): {len(rows)}   dclm tasks: {len(present)}/{len(tasks)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
