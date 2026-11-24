#!/usr/bin/env python3
"""Shared all-cluster progress report: CSV schema, merge and Markdown renderer.

``progress_tracker.py`` runs on one cluster at a time (it needs that cluster's
sacct and run directories).  With ``--accumulate`` it merges its rows into a
single CSV that holds every model size and every cluster (LUMI, LEO, MN5), and
renders the Markdown summary from that merged CSV via this module.

Merge rule (see :func:`merge_rows`): a tracker run on machine M for size S is
authoritative for M's jobs of S.  It replaces all of M's ``local`` rows of S,
replaces rows other machines had ``inferred`` for M on the runs M tracked now,
and adds its own ``inferred`` rows for other clusters only where nothing better
exists.  Rows of other machines / sizes are never touched, so running the
tracker once per cluster accumulates the full picture.

Regenerate the Markdown from the CSV alone (e.g. after resolving a git merge
conflict in the CSV)::

    python tools/progress_report.py progress.csv --md progress.md
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import tempfile
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path

_tools_dir = str(Path(__file__).parent)
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)

from write_guard import guard_write  # noqa: E402

MACHINES = ("LUMI", "LEO", "MN5")
TIER_SORT_ORDER = {"center": 0, "cross": 1, "diagonal": 2}

# Fixed, machine-independent column set.  Column names must never depend on
# the machine the tracker ran on, otherwise CSVs from different clusters
# cannot be merged.
GPU_H_COLS = tuple(f"GPU-h({m})" for m in MACHINES)
CSV_FIELDS = [
    "Size",
    "Run",
    "Machine",  # cluster the job ran on
    "Source",  # local: tracked on Machine itself; inferred: seen in synced logs elsewhere
    "JobID",
    "Owner",
    "StartTime",
    "Elapsed",
    "N_ne(B)",
    "N(B)",
    "D(B)",
    "C(10^18)",
    "Tier",
    "Stage",
    "TotIter",
    "CurIter",
    "LastCkpt",
    "Prog%",
    "TrainLoss",
    "ValLoss",
    "LR",
    "GBS",
    "MBS",
    "Nodes",
    "Workers",
    "TFLOP/s/GPU",
    "Tok/s/GPU",
    "TTFI(min)",
    "TTFI-GPU-h",
    "LowTP-time(h)",
    "LowTP-GPU-h",
    "Overhead-time(h)",
    "Overhead-GPU-h",
    *GPU_H_COLS,
    "GPU-h",
    "Overhead%",
    "Remaining-GPU-h",  # per run, on the run's latest job row of that machine
    "Ckpt-GB",  # per run, idem
    "Ckpt-GB-remaining",  # per run, idem
    "Emoji",
    "Status",
    "Action",
    "Error",
    "OutputPath",  # run directory on Machine (Megatron --save without /checkpoints)
    "DataPath",  # common root of the tokenized-data prefixes the job loaded
    "DataMixPath",  # Megatron --data-args-path (the data mix file)
    "Config",  # sweep config the tracker was run with
    "UpdatedAt",  # when the tracker run that wrote this row happened
]

_RE_SIZE_NUM = re.compile(r"^(\d+(?:\.\d+)?)B")


# ── CSV I/O ───────────────────────────────────────────────────────────────────


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a progress CSV.  Unknown columns are dropped and missing ones are
    filled with "", so files written by older schema versions still load."""
    with open(path, newline="") as f:
        return [{k: (row.get(k) or "") for k in CSV_FIELDS} for row in csv.DictReader(f)]


def _atomic_write(path: Path, write) -> None:
    guard_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="") as f:
            write(f)
        os.chmod(tmp, 0o664)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_csv(path: str | Path, rows: list[dict[str, str]]) -> None:
    def _w(f):
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in CSV_FIELDS})

    _atomic_write(Path(path), _w)


def write_text(path: str | Path, text: str) -> None:
    _atomic_write(Path(path), lambda f: f.write(text))


# ── Ordering ──────────────────────────────────────────────────────────────────


def size_sort_key(size: str) -> tuple:
    m = _RE_SIZE_NUM.match(size or "")
    return (0, float(m.group(1)), size) if m else (1, 0.0, size or "")


def run_sort_key(run: str, tier: str, stage: str) -> tuple:
    """Same ordering as the tracker's summary table: tier, lr, gbsz, stable
    before decays, token budget."""
    lr_m = re.search(r"_lr([0-9.e+\-]+)_", run)
    gbsz_m = re.search(r"_gbsz(\d+)_", run)
    tok_m = re.search(r"(?:stable|decay)(\d+)BT", run)
    try:
        lr = float(lr_m.group(1)) if lr_m else 0.0
    except ValueError:
        lr = 0.0
    return (
        TIER_SORT_ORDER.get(tier, 99),
        lr,
        int(gbsz_m.group(1)) if gbsz_m else 0,
        0 if stage == "stable" else 1,
        int(tok_m.group(1)) if tok_m else 0,
        run,
    )


def _machine_key(m: str) -> tuple:
    return (MACHINES.index(m), m) if m in MACHINES else (len(MACHINES), m)


def sort_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Deterministic order (keeps git diffs of the shared CSV small)."""
    return sorted(
        rows,
        key=lambda r: (
            size_sort_key(r.get("Size", "")),
            run_sort_key(r.get("Run", ""), r.get("Tier", ""), r.get("Stage", "")),
            _machine_key(r.get("Machine", "")),
            r.get("StartTime", "") or "",
            r.get("JobID", "").zfill(12),
        ),
    )


# ── Merge ─────────────────────────────────────────────────────────────────────


def merge_rows(
    existing: list[dict[str, str]],
    new: list[dict[str, str]],
    *,
    machine: str,
    size: str,
    tracked_runs: set[str],
    carry_ckpt: bool = True,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Merge one tracker run (*new*, from *machine* for *size*) into *existing*.

    1. Drop existing rows of (machine, size) that are ``local``, and the
       ``inferred`` ones for runs this tracker run covered.  Those are
       superseded by what the machine itself just saw.
    2. Add the new ``local`` rows.  A job already present as another machine's
       ``local`` row with the same (Run, JobID) is skipped: that machine's
       sacct vouched for it, so a same-ID hit here is an ID-space collision
       the GPU-count heuristic missed.
    3. Add the new ``inferred`` rows only where no row with the same
       (Run, JobID) remains, so the real machine's rows always win.
    4. Without --compute-storage, carry the previous Ckpt-GB values of
       (machine, run) forward instead of blanking them.
    """
    stats = Counter()
    kept: list[dict[str, str]] = []
    dropped_ckpt: dict[str, tuple[str, str]] = {}
    for r in existing:
        if r.get("Machine") == machine and r.get("Size") == size and (
            r.get("Source", "local") != "inferred" or r.get("Run") in tracked_runs
        ):
            if r.get("Ckpt-GB") or r.get("Ckpt-GB-remaining"):
                dropped_ckpt[r["Run"]] = (r.get("Ckpt-GB", ""), r.get("Ckpt-GB-remaining", ""))
            stats["replaced"] += 1
            continue
        kept.append(r)

    other_local_jobs = {
        (r["Run"], r["JobID"])
        for r in kept
        if r.get("JobID") and r.get("Source", "local") != "inferred"
    }
    out = list(kept)
    present = {(r["Run"], r["JobID"]) for r in out if r.get("JobID")}
    for r in new:
        key = (r["Run"], r["JobID"])
        if r.get("Source") == "inferred":
            if r["JobID"] and key in present:
                stats["inferred_skipped"] += 1
                continue
            stats["inferred_added"] += 1
            present.add(key)
        else:
            if r["JobID"] and key in other_local_jobs:
                print(
                    f"  [merge] skip {r['Run']} job {r['JobID']}: already a local job of "
                    f"another machine (Slurm ID collision)",
                    file=sys.stderr,
                )
                stats["collision_skipped"] += 1
                continue
            stats["local_added"] += 1
        out.append(r)

    if carry_ckpt and dropped_ckpt:
        latest_new: dict[str, dict[str, str]] = {}
        for r in out:
            if r.get("Machine") == machine and r.get("Size") == size and r.get("Source") == "local":
                latest_new[r["Run"]] = r  # tracker order: last row of a run is its latest job
        for run, (gb, rem) in dropped_ckpt.items():
            r = latest_new.get(run)
            if r is not None and not r.get("Ckpt-GB") and not r.get("Ckpt-GB-remaining"):
                r["Ckpt-GB"], r["Ckpt-GB-remaining"] = gb, rem
    return sort_rows(out), dict(stats)


# ── Aggregation per run ───────────────────────────────────────────────────────


def _f(s: str | None) -> float | None:
    if s is None:
        return None
    s = str(s).strip().lstrip("~")
    if not s or s in ("N/A", "--", "—"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _sum(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    return sum(vals) if vals else None


# Job-less rows: the most informative one across clusters wins.
_PLACEHOLDER_RANK = {"QUEUED": 0, "CKPT_ONLY": 1, "CANCELLED": 2, "NOT_LAUNCHED": 3}
_PLACEHOLDER_EMOJI = {"QUEUED": "🕒", "CKPT_ONLY": "📦", "CANCELLED": "🚫", "NOT_LAUNCHED": "⚪"}


def _start_key(r: dict[str, str]) -> str:
    # A job without a start time is still pending, i.e. the newest one.
    return r.get("StartTime") or "9999"


def aggregate_run(rows: list[dict[str, str]]) -> dict:
    """Collapse all rows of one run (any machines) into one summary record."""
    jobs = [r for r in rows if r.get("JobID")]
    first = rows[0]
    agg: dict = {
        "run": first["Run"],
        "tier": next((r["Tier"] for r in rows if r.get("Tier")), ""),
        "stage": next((r["Stage"] for r in rows if r.get("Stage")), ""),
    }
    machines = sorted({r["Machine"] for r in jobs if r.get("Machine")}, key=_machine_key)
    agg["clusters"] = "+".join(machines)
    agg["machines"] = machines

    progs = [_f(r.get("Prog%")) for r in rows]
    progs = [p for p in progs if p is not None]
    agg["progress"] = max(progs) if progs else None

    if jobs:
        latest = max(jobs, key=lambda r: (_start_key(r), r.get("JobID", "").zfill(12)))
        if any(r.get("Status") == "DONE" for r in jobs):
            agg["emoji"], agg["status"] = "✅", "DONE"
        else:
            agg["emoji"], agg["status"] = latest.get("Emoji", ""), latest.get("Status", "")
        ref_machine = latest.get("Machine", "")
    else:
        best = min(rows, key=lambda r: _PLACEHOLDER_RANK.get(r.get("Status", ""), 9))
        agg["status"] = best.get("Status", "") or "NOT_LAUNCHED"
        agg["emoji"] = best.get("Emoji") or _PLACEHOLDER_EMOJI.get(agg["status"], "")
        ref_machine = best.get("Machine", "")

    agg["gpu_h_by_machine"] = {
        m: _sum(_f(r.get("GPU-h")) for r in jobs if r.get("Machine") == m) for m in MACHINES
    }
    agg["gpu_h"] = _sum(_f(r.get("GPU-h")) for r in jobs)
    agg["ttfi_gpu_h"] = _sum(_f(r.get("TTFI-GPU-h")) for r in jobs)
    agg["lowtp_gpu_h"] = _sum(_f(r.get("LowTP-GPU-h")) for r in jobs)
    agg["overhead_gpu_h"] = _sum(
        _f(r.get("Overhead-GPU-h")) for r in jobs if _f(r.get("GPU-h")) is not None
    )
    agg["overhead_pct"] = (
        agg["overhead_gpu_h"] / agg["gpu_h"] * 100.0
        if agg["overhead_gpu_h"] is not None and agg["gpu_h"]
        else None
    )

    # Run-level estimates live on one row per (run, machine).  Prefer the
    # machine that holds the run's latest job; otherwise any machine.
    def _pick(col: str) -> str:
        cands = [r for r in rows if r.get(col)]
        if not cands:
            return ""
        same = [r for r in cands if r.get("Machine") == ref_machine]
        return max(same or cands, key=lambda r: (r.get("UpdatedAt", ""), _start_key(r)))[col]

    if agg["status"] == "DONE":
        agg["remaining"], agg["remaining_est"] = 0.0, False
    else:
        rem = _pick("Remaining-GPU-h")
        agg["remaining"], agg["remaining_est"] = _f(rem), rem.startswith("~")
    # Checkpoints exist separately on every cluster: sum the per-machine sizes.
    per_machine_ckpt = {}
    for r in rows:
        v = _f(r.get("Ckpt-GB"))
        if v is not None:
            per_machine_ckpt[r.get("Machine", "")] = v
    agg["ckpt_gb"] = sum(per_machine_ckpt.values()) if per_machine_ckpt else None
    agg["ckpt_rem_gb"] = _f(_pick("Ckpt-GB-remaining"))

    # Locations per machine (distinct values, newest first)
    locs: "OrderedDict[str, dict[str, list[str]]]" = OrderedDict()
    for m in machines:
        mj = sorted(
            (r for r in jobs if r.get("Machine") == m), key=_start_key, reverse=True
        )
        d: dict[str, list[str]] = {}
        for col in ("OutputPath", "DataMixPath", "DataPath"):
            seen: list[str] = []
            for r in mj:
                v = r.get(col, "")
                if v and v not in seen:
                    seen.append(v)
            d[col] = seen
        d["inferred"] = all(r.get("Source") == "inferred" for r in mj)
        locs[m] = d
    agg["locations"] = locs
    return agg


# ── Markdown ──────────────────────────────────────────────────────────────────


def _n(v: float | None, fmt: str = ".1f", na: str = "N/A") -> str:
    return format(v, fmt) if v is not None else na


def _rem_str(v: float | None, est: bool) -> str:
    if v is None:
        return "N/A"
    return f"~{v:.1f}" if est else f"{v:.1f}"


def _code_list(vals: list[str]) -> str:
    return "<br>".join(f"`{v}`" for v in vals) if vals else "—"


_SUMMARY_HDR = (
    ["T#", "#", "Experiment", "Tier", "Progress", "Status", "Clusters"]
    + ["TTFI-GPU-h", "LowTP-GPU-h", "Overhead-GPU-h"]
    + list(GPU_H_COLS)
    + ["GPU-h", "Overhead%", "Remaining-GPU-h", "Ckpt-GB", "Ckpt-GB-remaining"]
)


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |\n"


class _Totals:
    def __init__(self) -> None:
        self.ttfi = self.lowtp = self.oh = self.gpu_h = self.rem = 0.0
        self.ckpt = self.ckpt_rem = 0.0
        self.by_machine = {m: 0.0 for m in MACHINES}
        self.has = Counter()

    def add(self, a: dict) -> None:
        for attr, key in (
            ("ttfi", "ttfi_gpu_h"),
            ("lowtp", "lowtp_gpu_h"),
            ("oh", "overhead_gpu_h"),
            ("gpu_h", "gpu_h"),
            ("rem", "remaining"),
            ("ckpt", "ckpt_gb"),
            ("ckpt_rem", "ckpt_rem_gb"),
        ):
            if a.get(key) is not None:
                setattr(self, attr, getattr(self, attr) + a[key])
                self.has[attr] += 1
        for m, v in a["gpu_h_by_machine"].items():
            if v is not None:
                self.by_machine[m] += v
                self.has[m] += 1

    def cells(self, label: str, tier: str = "") -> list[str]:
        def b(attr: str, v: float, fmt: str = ".1f") -> str:
            return f"**{v:{fmt}}**" if self.has[attr] else "**—**"

        oh_pct = self.oh / self.gpu_h * 100.0 if self.has["oh"] and self.gpu_h else None
        return (
            ["", "", f"**{label}**", f"**{tier}**" if tier else "", "", "", ""]
            + [b("ttfi", self.ttfi, ".2f"), b("lowtp", self.lowtp, ".2f"), b("oh", self.oh, ".2f")]
            + [b(m, self.by_machine[m]) for m in MACHINES]
            + [b("gpu_h", self.gpu_h)]
            + [f"**{oh_pct:.1f}%**" if oh_pct is not None else "**—**"]
            + [b("rem", self.rem), b("ckpt", self.ckpt), b("ckpt_rem", self.ckpt_rem)]
        )


def _group_runs(rows: list[dict[str, str]]) -> "OrderedDict[str, list[dict]]":
    """{size: [aggregated run, ...]} in display order."""
    by_size: dict[str, dict[str, list[dict[str, str]]]] = {}
    for r in rows:
        by_size.setdefault(r.get("Size", ""), OrderedDict()).setdefault(r["Run"], []).append(r)
    out: "OrderedDict[str, list[dict]]" = OrderedDict()
    for size in sorted(by_size, key=size_sort_key):
        aggs = [aggregate_run(rs) for rs in by_size[size].values()]
        aggs.sort(key=lambda a: run_sort_key(a["run"], a["tier"], a["stage"]))
        out[size] = aggs
    return out


def render_md(rows: list[dict[str, str]], *, now: str | None = None) -> str:
    now = now or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    grouped = _group_runs(rows)
    f: list[str] = []
    f.append(f"_Rendered: {now}_\n\n")
    f.append("# Training progress — all clusters\n\n")

    # Tracker freshness per machine
    f.append("## Last tracker run per cluster\n\n")
    f.append("| Cluster | Last tracker run | Sizes tracked |\n| --- | --- | --- |\n")
    for m in MACHINES:
        loc = [r for r in rows if r.get("Machine") == m and r.get("Source", "local") == "local"]
        ts = max((r.get("UpdatedAt", "") for r in loc), default="")
        sizes = sorted({r.get("Size", "") for r in loc if r.get("Size")}, key=size_sort_key)
        f.append(f"| {m} | {ts or 'never'} | {', '.join(sizes) or '—'} |\n")
    f.append(
        "\n_Each cluster's rows are only as fresh as its last tracker run. "
        "`inferred` jobs were seen in logs synced to another cluster and are "
        "replaced once the tracker runs on the cluster they ran on._\n\n"
    )

    # Overview across sizes
    f.append("## Overview\n\n")
    f.append(
        _md_row(["Size", "Runs", "DONE", "TRAINING", *GPU_H_COLS, "GPU-h", "Remaining-GPU-h"])
    )
    f.append(_md_row(["---"] * (6 + len(MACHINES))))
    grand = _Totals()
    for size, aggs in grouped.items():
        t = _Totals()
        for a in aggs:
            t.add(a)
            grand.add(a)
        st = Counter(a["status"] for a in aggs)
        f.append(
            _md_row(
                [size or "?", str(len(aggs)), str(st["DONE"]), str(st["TRAINING"])]
                + [f"{t.by_machine[m]:.1f}" if t.has[m] else "—" for m in MACHINES]
                + [f"{t.gpu_h:.1f}", f"{t.rem:.1f}" if t.has["rem"] else "—"]
            )
        )
    n_runs = sum(len(a) for a in grouped.values())
    f.append(
        _md_row(
            ["**TOTAL**", f"**{n_runs}**", "", ""]
            + [f"**{grand.by_machine[m]:.1f}**" if grand.has[m] else "**—**" for m in MACHINES]
            + [f"**{grand.gpu_h:.1f}**", f"**{grand.rem:.1f}**"]
        )
    )
    f.append("\n")

    for size, aggs in grouped.items():
        size_rows = [r for r in rows if r.get("Size", "") == size]
        f.append(f"## {size or 'Unknown size'}\n\n")

        # Sources
        f.append("### Sources\n\n")
        f.append("| Cluster | Last tracker run | Config | Results root |\n| --- | --- | --- | --- |\n")
        for m in MACHINES:
            loc = [r for r in size_rows if r.get("Machine") == m and r.get("Source", "local") == "local"]
            if not loc:
                continue
            ts = max(r.get("UpdatedAt", "") for r in loc)
            cfgs = sorted({r["Config"] for r in loc if r.get("Config")})
            outs = {r["OutputPath"] for r in loc if r.get("OutputPath")}
            try:
                root = os.path.commonpath(sorted(outs)) if len(outs) > 1 else (next(iter(outs)) if outs else "")
            except ValueError:
                root = ""
            f.append(f"| {m} | {ts} | {_code_list(cfgs)} | {_code_list([root] if root else [])} |\n")
        f.append("\n")

        # Summary table
        f.append("### Training progress summary\n\n")
        f.append(
            "_GPU-h per cluster is billed GPU-hours from that cluster's sacct "
            "(log-inferred for `inferred` jobs). ~ = remaining estimated from "
            "stable-run throughput (run not started)._\n\n"
        )
        f.append(_md_row(_SUMMARY_HDR))
        f.append(_md_row(["---"] * len(_SUMMARY_HDR)))
        sep = _md_row(["—"] * len(_SUMMARY_HDR))
        tier_tot = _Totals()
        size_tot = _Totals()
        tier_idx = 0
        for idx, a in enumerate(aggs, start=1):
            prev_tier = aggs[idx - 2]["tier"] if idx > 1 else None
            tier_idx = 1 if a["tier"] != prev_tier else tier_idx + 1
            prog = f"{a['progress']:.1f}%" if a["progress"] is not None else "N/A"
            f.append(
                _md_row(
                    [str(tier_idx), str(idx), a["run"], a["tier"], prog, f"{a['emoji']} {a['status']}".strip()]
                    + [a["clusters"] or ""]
                    + [_n(a["ttfi_gpu_h"], ".2f"), _n(a["lowtp_gpu_h"], ".2f"), _n(a["overhead_gpu_h"], ".2f")]
                    + [_n(a["gpu_h_by_machine"][m], ".1f", "—") for m in MACHINES]
                    + [_n(a["gpu_h"])]
                    + [f"{a['overhead_pct']:.1f}%" if a["overhead_pct"] is not None else "N/A"]
                    + [_rem_str(a["remaining"], a["remaining_est"])]
                    + [_n(a["ckpt_gb"], ".1f", "—"), _n(a["ckpt_rem_gb"], ".1f", "—")]
                )
            )
            tier_tot.add(a)
            size_tot.add(a)
            is_last = idx == len(aggs)
            if is_last or aggs[idx]["tier"] != a["tier"]:
                if a["tier"]:
                    f.append(_md_row(tier_tot.cells(f"[{a['tier']} total]", a["tier"])))
                    f.append(sep)
                tier_tot = _Totals()
        f.append(_md_row(size_tot.cells("TOTAL")))
        f.append("\n")

        # Locations
        f.append("### Run locations\n\n")
        f.append("| Experiment | Cluster | OutputPath | DataMixPath | DataPath |\n| --- | --- | --- | --- | --- |\n")
        for a in aggs:
            for m, d in a["locations"].items():
                tag = f"{m} (inferred)" if d["inferred"] else m
                f.append(
                    f"| {a['run']} | {tag} | {_code_list(d['OutputPath'])} | "
                    f"{_code_list(d['DataMixPath'])} | {_code_list(d['DataPath'])} |\n"
                )
        f.append("\n")

        # Status breakdown (per run, aggregated across clusters)
        f.append("### Status breakdown (runs)\n\n")
        for status, cnt in sorted(Counter(a["status"] for a in aggs).items()):
            f.append(f"- **{status}**: {cnt}\n")
        f.append("\n")
    return "".join(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the all-cluster progress Markdown from progress.csv.")
    ap.add_argument("csv", help="Merged progress CSV (written by progress_tracker.py --accumulate)")
    ap.add_argument("--md", required=True, help="Markdown output path")
    args = ap.parse_args()
    rows = read_csv(args.csv)
    write_text(args.md, render_md(rows))
    print(f"Wrote {args.md} from {len(rows)} rows")


if __name__ == "__main__":
    main()
