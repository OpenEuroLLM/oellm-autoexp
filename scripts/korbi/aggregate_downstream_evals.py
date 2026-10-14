#!/usr/bin/env python3
"""Aggregate oellm-eval downstream results into a CSV + a markdown report.

The downstream sweep leaves one lm_eval results JSON per invocation, at a
RANDOM filename (``oellm/resources/template.sbatch`` passes
``--output_path <evals_dir>/$(openssl rand -hex 5).json``). Reruns therefore
add duplicates rather than overwriting, and nothing downstream de-duplicates
them. This script does:

  1. walk the downstream tree and parse every results JSON
  2. identify the model from the JSON itself (``model_name``, falling back to
     ``config.model_args.pretrained``) -- NOT from the path, so a moved or
     re-nested tree still attributes correctly
  3. de-duplicate on (model, task, n_shot), keeping the value from the
     most-recently-written JSON (files are walked oldest-first, so a later run
     overwrites an earlier one). Disagreements are still counted and reported;
     they are resolved in favour of the newest result rather than left to
     directory iteration order.
     n_shot is part of the key because the same task is legitimately evaluated
     at several shot counts (hellaswag runs at both 0-shot and 10-shot here);
     keying on (model, task) alone silently discards one of them and reports
     the pair as a conflict.
  4. parse the run name into (variant, size, gbs, lr, budget)
  5. emit a long-format CSV and a markdown report

Usage
-----
    python scripts/korbi/aggregate_downstream_evals.py \\
        --root /e/scratch/.../architecture_scaling_variants_multilingual_main_7to1_downstream \\
        --out-prefix dump/downstream

writes ``dump/downstream.csv`` and ``dump/downstream.md``.

Options:
    --root PATH        downstream group dir (repeatable)
    --out-prefix P     output prefix (default: dump/downstream_evals)
    --variants a,b     restrict to these variants
    --min-tasks N      hide (variant,size,budget) cells with fewer than N tasks
    --quiet            suppress per-file warnings
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import re
import sys

# qwen3_<variant>_<size>_ne_lr<lr>_gbsz<gbs>_<stage-ish>decay<B>BT
RUN_RE = re.compile(
    r"qwen3_(?P<variant>.+?)_(?P<size>\d\.\d+B)_ne_lr(?P<lr>[\d.]+)_gbsz(?P<gbs>\d+)"
    r"(?:_(?P<stage>[a-z]+))?_?decay(?P<budget>\d+)BT"
)

# Preference order for the headline metric of a task. lm-eval keys carry a
# filter suffix (``acc,none``), so match on the part before the comma.
METRIC_PREFERENCE = ("acc_norm", "acc", "exact_match", "f1", "bleu", "chrf")


def parse_run_name(model_path: str) -> dict[str, str] | None:
    m = RUN_RE.search(pathlib.Path(model_path).name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "variant": d["variant"],
        "size": d["size"],
        "lr": d["lr"],
        "gbs": d["gbs"],
        "budget_bt": d["budget"],
    }


def pick_metric(metrics: dict) -> tuple[str | None, float | None, float | None]:
    """Return (metric_name, value, stderr) using METRIC_PREFERENCE."""
    plain = {}
    stderr = {}
    for key, val in (metrics or {}).items():
        if not isinstance(key, str):
            continue
        base = key.split(",", 1)[0]
        if base.endswith("_stderr"):
            stderr[base[: -len("_stderr")]] = val
        elif isinstance(val, (int, float)):
            plain.setdefault(base, val)
    for name in METRIC_PREFERENCE:
        if name in plain:
            return name, float(plain[name]), _as_float(stderr.get(name))
    for name, val in plain.items():  # fall back to whatever numeric exists
        return name, float(val), _as_float(stderr.get(name))
    return None, None, None


def _as_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def collect(roots: list[pathlib.Path], quiet: bool) -> tuple[list[dict], dict]:
    seen: dict[tuple[str, str], dict] = {}
    stats = collections.Counter()
    conflicts: list[str] = []

    # Oldest first, so that on a duplicate key the LATER file wins. Ordering by
    # mtime (not by name) is what makes "newest" meaningful: lm_eval writes a
    # random hex filename, so lexical order carries no chronology at all.
    files = []
    for root in roots:
        files.extend(root.rglob("*.json"))
    files.sort(key=lambda p: p.stat().st_mtime)
    stats["files"] = len(files)

    for path in files:
        try:
            doc = json.loads(path.read_text())
        except Exception:
            stats["unparseable"] += 1
            continue
        results = doc.get("results")
        if not isinstance(results, dict):
            stats["no_results"] += 1
            continue
        model_path = doc.get("model_name")
        if not model_path:
            margs = (doc.get("config") or {}).get("model_args") or {}
            # model_args is a DICT in this lm-eval version, not a string.
            model_path = margs.get("pretrained") if isinstance(margs, dict) else None
        if not model_path:
            stats["no_model"] += 1
            continue
        model = pathlib.Path(model_path).name
        groups = set(doc.get("group_subtasks") or {})
        nshot_map = doc.get("n-shot") or {}
        # Stage comes from the run directory (…_evaldclm / …_evalml). Both stages
        # evaluate the SAME model dir, so the model name cannot distinguish them
        # -- without this, a DCLM score and a multilingual score are averaged
        # together and the mean depends on which stages happen to have run.
        stage = ""
        for part in path.parts[::-1]:
            if part.endswith("_evaldclm") or part.endswith("_evalml"):
                stage = part.rsplit("_", 1)[-1]
                break

        for task, metrics in results.items():
            name, value, err = pick_metric(metrics)
            if value is None:
                stats["no_metric"] += 1
                continue
            nshot = nshot_map.get(task)
            key = (model, task, nshot)
            row = {
                "model": model,
                "task": task,
                "n_shot": nshot,
                "task_key": f"{task}@{nshot}shot" if nshot is not None else task,
                "stage": stage,
                "is_group": task in groups,
                "metric": name,
                "value": value,
                "stderr": err,
                "source": str(path),
            }
            if key in seen:
                stats["duplicate"] += 1
                prev = seen[key]["value"]
                if abs(prev - value) > 1e-9:
                    stats["conflict"] += 1
                    conflicts.append(
                        f"{model} | {task} @{nshot}-shot: {prev} (older) -> {value} (newer, kept)"
                    )
                seen[key] = row  # newest wins
                continue
            seen[key] = row
            stats["unique"] += 1

    rows = []
    for row in seen.values():
        meta = parse_run_name(row["model"])
        if meta is None:
            stats["unparsed_run_name"] += 1
            if not quiet:
                print(f"  ! could not parse run name: {row['model']}", file=sys.stderr)
            continue
        rows.append({**meta, **row})
    stats["conflicts_list"] = conflicts
    return rows, stats


def write_csv(rows: list[dict], path: pathlib.Path) -> None:
    cols = [
        "variant",
        "size",
        "gbs",
        "lr",
        "budget_bt",
        "stage",
        "task",
        "n_shot",
        "is_group",
        "metric",
        "value",
        "stderr",
        "model",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(
            rows,
            key=lambda r: (
                r["variant"],
                r["size"],
                int(r["budget_bt"]),
                r["task"],
                r["n_shot"] if r["n_shot"] is not None else -1,
            ),
        ):
            w.writerow(r)


def _fmt(v: float | None, nd: int = 4) -> str:
    return "--" if v is None else f"{v:.{nd}f}"


def write_markdown(rows: list[dict], stats: dict, path: pathlib.Path, min_tasks: int) -> None:
    leaf = [r for r in rows if not r["is_group"]]
    variants = sorted({r["variant"] for r in rows})
    out: list[str] = []
    out.append("# Downstream evaluation results\n")
    out.append(
        f"Parsed **{stats['files']}** result JSONs into **{stats['unique']}** unique "
        f"(model, task) pairs; **{stats['duplicate']}** duplicate evaluations were "
        f"discarded.\n"
    )
    if stats["conflict"]:
        out.append(
            f"> **NOTE: {stats['conflict']} duplicated pairs disagree on their score.**\n>\n"
            "> Resolved by keeping the MOST RECENT result for each "
            "(model, task, n_shot). Listed below as `older -> newer (kept)` so a "
            "disagreement is still visible rather than silently absorbed.\n"
        )
        for line in stats["conflicts_list"][:20]:
            out.append(f"> - {line}\n")
    else:
        out.append(
            "All duplicated pairs agreed exactly, so de-duplication is lossless and the "
            "aggregates below are independent of which copy was kept.\n"
        )

    # ---- coverage -------------------------------------------------------
    out.append("\n## Coverage\n")
    cov = collections.defaultdict(set)
    for r in leaf:
        cov[(r["variant"], r["size"])].add(r["budget_bt"])
    sizes = sorted({r["size"] for r in leaf})
    out.append("| variant | " + " | ".join(sizes) + " |")
    out.append("|---" * (len(sizes) + 1) + "|")
    for v in variants:
        cells = [str(len(cov.get((v, s), ()))) for s in sizes]
        out.append(f"| `{v}` | " + " | ".join(cells) + " |")
    out.append("\n_budget points present per (variant, size)._\n")

    # ---- headline: macro-average over the COMMON task set ----------------
    # Averaging over whatever each variant happens to have evaluated compares
    # different task sets and produces differences that look architectural but
    # are not: an arm that also ran the multilingual stage averages over ~1183
    # tasks (including hard generative ones) while an arm part-way through the
    # DCLM stage averages over a handful of easy ones. Every cell in a row is
    # therefore restricted to the tasks ALL present variants share, and the
    # per-variant task counts are printed so partial coverage is visible.
    out.append("\n## Macro-average over the common task set\n")
    byv: dict[tuple, dict[str, float]] = collections.defaultdict(dict)
    for r in leaf:
        byv[(r["variant"], r["size"], int(r["budget_bt"]))][r["task_key"]] = r["value"]
    cells = sorted({(k[1], k[2]) for k in byv})
    out.append(
        "| size | budget (BT) | "
        + " | ".join(f"`{v}`" for v in variants)
        + " | n_common | per-variant n_tasks |"
    )
    out.append("|---" * (len(variants) + 4) + "|")
    for size, budget in cells:
        present = [v for v in variants if byv.get((v, size, budget))]
        if not present:
            continue
        common = set.intersection(*(set(byv[(v, size, budget)]) for v in present))
        if len(common) < min_tasks:
            continue
        vals = []
        for v in variants:
            d = byv.get((v, size, budget))
            vals.append(_fmt(sum(d[t] for t in common) / len(common)) if d else "--")
        counts = ",".join(f"{v.split('_')[0]}:{len(byv[(v, size, budget)])}" for v in present)
        out.append(f"| {size} | {budget} | " + " | ".join(vals) + f" | {len(common)} | {counts} |")
    out.append(
        "\n_Unweighted mean over the tasks shared by every variant in that row, so the "
        "columns are directly comparable. `n_common` is the size of that shared set; "
        "`per-variant n_tasks` shows how many tasks each variant actually has, so a row "
        "whose intersection is much smaller than the individual counts is a partially "
        "evaluated row, not a real difference._\n"
    )

    # ---- per-task, largest budget per size -------------------------------
    out.append("\n## Per-task, largest budget per size\n")
    for size in sizes:
        budgets = [b for (v, s, b) in byv if s == size]
        if not budgets:
            continue
        big = max(budgets)
        tasks = sorted(
            {r["task_key"] for r in leaf if r["size"] == size and int(r["budget_bt"]) == big}
        )
        if not tasks:
            continue
        out.append(f"\n### {size} @ {big} BT\n")
        out.append("| task | " + " | ".join(f"`{v}`" for v in variants) + " |")
        out.append("|---" * (len(variants) + 1) + "|")
        idx = {
            (r["variant"], r["task_key"]): r
            for r in leaf
            if r["size"] == size and int(r["budget_bt"]) == big
        }
        for t in tasks:
            cells = [_fmt(idx[(v, t)]["value"]) if (v, t) in idx else "--" for v in variants]
            out.append(f"| {t} | " + " | ".join(cells) + " |")

    missing = [
        k
        for k in ("unparseable", "no_results", "no_model", "no_metric", "unparsed_run_name")
        if stats.get(k)
    ]
    if missing:
        out.append("\n## Skipped inputs\n")
        for k in missing:
            out.append(f"- `{k}`: {stats[k]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--root",
        action="append",
        required=True,
        type=pathlib.Path,
        help="downstream group directory (repeatable)",
    )
    p.add_argument("--out-prefix", default="dump/downstream_evals")
    p.add_argument("--variants", default=None, help="comma-separated allow-list")
    p.add_argument("--min-tasks", type=int, default=1)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    for root in args.root:
        if not root.exists():
            print(f"ERROR: no such directory: {root}", file=sys.stderr)
            return 1

    rows, stats = collect(args.root, args.quiet)
    if args.variants:
        keep = {v.strip() for v in args.variants.split(",")}
        rows = [r for r in rows if r["variant"] in keep]
    if not rows:
        print("ERROR: no parseable results found", file=sys.stderr)
        return 1

    csv_path = pathlib.Path(f"{args.out_prefix}.csv")
    md_path = pathlib.Path(f"{args.out_prefix}.md")
    write_csv(rows, csv_path)
    write_markdown(rows, stats, md_path, args.min_tasks)

    print(
        f"files={stats['files']} unique={stats['unique']} "
        f"duplicates_dropped={stats['duplicate']} conflicts={stats['conflict']}"
    )
    print(f"rows={len(rows)}  variants={sorted({r['variant'] for r in rows})}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    if stats["conflict"]:
        print(
            f"WARNING: {stats['conflict']} duplicate pairs disagree -- see the report",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
