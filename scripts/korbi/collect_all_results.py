#!/usr/bin/env python3
"""Collect EVERY result for the architecture sweep into one wide CSV.

Two result families live in different places and are joined here on the run
identity ``(variant, size, gbs, lr, budget_bt)``:

  * **validation loss / PPL** -- produced by the Megatron-native eval sweeps
    (``eval_*_scaling_all.yaml``) and scraped by
    ``scripts/korbi/scaling_laws/refresh_metrics.sh`` into a long CSV.
  * **downstream task scores** -- lm-eval JSONs under the downstream output
    group, parsed (and de-duplicated on ``(model, task, n_shot)``) by
    ``aggregate_downstream_evals.py``, which this script imports rather than
    reimplements.

Output: one row per run, with ``val_loss``/``val_ppl``, a macro-average over the
downstream tasks, and one column per ``task@Nshot``. That is the shape you want
for plotting or a spreadsheet; the long format stays available from
``aggregate_downstream_evals.py``.

Only the COOLDOWN endpoints are joined: downstream evaluates
``firstcd_decay<B>BT`` checkpoints, which correspond to val rows of type
``evalcd``. Stable-branch val rows (``evalstable``) have no downstream
counterpart and are emitted with empty task columns unless --cooldown-only.

Usage
-----
    python scripts/korbi/collect_all_results.py \\
        --downstream-root /e/scratch/.../..._downstream \\
        --val-csv dump/eval_val_metrics.csv \\
        --out dump/all_mlscaling_results.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import pathlib
import statistics
import sys

_HERE = pathlib.Path(__file__).resolve().parent


def _load_aggregator():
    """Import aggregate_downstream_evals.py as a module (same directory)."""
    path = _HERE / "aggregate_downstream_evals.py"
    spec = importlib.util.spec_from_file_location("agg_downstream", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_val_csv(path: pathlib.Path) -> dict[tuple, dict]:
    """Long val CSV -> {(variant,size,gbs,lr,budget,type): {val_loss,
    val_ppl}}."""
    out: dict[tuple, dict] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        for row in csv.DictReader(fh):
            try:
                key = (
                    row["variant"],
                    row["size"],
                    str(int(row["gbs"])),
                    _norm_lr(row["lr"]),
                    str(int(row["budgetBT"])),
                    row["type"],
                )
            except (ValueError, KeyError):
                continue
            out[key] = {
                "val_loss": _f(row.get("val_loss")),
                "val_ppl": _f(row.get("val_ppl")),
                "iter": row.get("iter", ""),
            }
    return out


def _norm_lr(v: str) -> str:
    """Lr appears as 0.002 / 0.0005 in both sources; normalise for joining."""
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return str(v)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--downstream-root", action="append", required=True, type=pathlib.Path)
    p.add_argument(
        "--val-csv", type=pathlib.Path, default=pathlib.Path("dump/eval_val_metrics.csv")
    )
    p.add_argument(
        "--out", type=pathlib.Path, default=pathlib.Path("dump/all_mlscaling_results.csv")
    )
    p.add_argument(
        "--cooldown-only",
        action="store_true",
        help="emit only evalcd rows (those with a downstream counterpart)",
    )
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    agg = _load_aggregator()
    rows, stats = agg.collect(args.downstream_root, args.quiet)
    if stats.get("conflict"):
        print(
            f"WARNING: {stats['conflict']} duplicated (model,task,n_shot) pairs disagree; "
            "resolved by keeping the most recent result",
            file=sys.stderr,
        )

    # downstream: {(variant,size,gbs,lr,budget): {task@Nshot: value}}, plus the
    # same split by stage so the summary means stay comparable between rows.
    ds: dict[tuple, dict[str, float]] = {}
    by_stage: dict[tuple, dict[str, dict[str, float]]] = {}
    for r in rows:
        if r["is_group"]:
            continue
        key = (
            r["variant"],
            r["size"],
            str(int(r["gbs"])),
            _norm_lr(r["lr"]),
            str(int(r["budget_bt"])),
        )
        ds.setdefault(key, {})[r["task_key"]] = r["value"]
        st = r.get("stage") or "unknown"
        by_stage.setdefault(key, {}).setdefault(st, {})[r["task_key"]] = r["value"]

    val = read_val_csv(args.val_csv)
    tasks = sorted({t for d in ds.values() for t in d})

    # Union of run keys from both sources, so a run missing one side still appears.
    keys: set[tuple] = set()
    for k in ds:
        keys.add(k + ("evalcd",))
    for k in val:
        keys.add(k)
    if args.cooldown_only:
        keys = {k for k in keys if k[5] == "evalcd"}

    # dclm_mean / ml_mean are reported SEPARATELY and never combined: a row that
    # also ran the multilingual stage averages over ~1184 tasks while a
    # DCLM-only row averages over 19, and one pooled mean across those is not a
    # comparable number.
    cols = [
        "variant",
        "size",
        "gbs",
        "lr",
        "budget_bt",
        "eval_type",
        "val_loss",
        "val_ppl",
        "n_dclm_tasks",
        "dclm_mean",
        "n_ml_tasks",
        "ml_mean",
    ] + tasks
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_both = 0
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for k in sorted(keys, key=lambda k: (k[0], k[1], int(k[4]), k[5])):
            variant, size, gbs, lr, budget, typ = k
            d = ds.get((variant, size, gbs, lr, budget), {}) if typ == "evalcd" else {}
            v = val.get(k, {})
            if d and v:
                n_both += 1
            st = by_stage.get((variant, size, gbs, lr, budget), {}) if typ == "evalcd" else {}
            dclm = st.get("evaldclm", {})
            ml = st.get("evalml", {})
            rec = {
                "variant": variant,
                "size": size,
                "gbs": gbs,
                "lr": lr,
                "budget_bt": budget,
                "eval_type": typ,
                "val_loss": v.get("val_loss"),
                "val_ppl": v.get("val_ppl"),
                "n_dclm_tasks": len(dclm) or None,
                "dclm_mean": round(statistics.mean(dclm.values()), 6) if dclm else None,
                "n_ml_tasks": len(ml) or None,
                "ml_mean": round(statistics.mean(ml.values()), 6) if ml else None,
            }
            rec.update({t: d.get(t) for t in tasks})
            w.writerow(rec)

    print(f"downstream runs: {len(ds)}   val rows: {len(val)}   task columns: {len(tasks)}")
    print(f"rows written: {len(keys)}   rows with BOTH val loss and downstream: {n_both}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
