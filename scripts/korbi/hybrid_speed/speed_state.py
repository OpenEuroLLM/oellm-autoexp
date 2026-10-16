"""The current speed state of all hybrid speed-suite runs, as one markdown file.

    python scripts/korbi/hybrid_speed/speed_state.py $OUTPUT_DIR/hybrid_speed -o SPEED_STATE.md \
        [--notes config/experiments/korbi/hybrid_speed/NOTES.md] [--loop 600]

Per family (dense130M, moe120M1200M, ...): the LATEST measurement of every arm across all
suites of that family (suite directories ``<suite>_<family>``, the newest run of an arm
wins), relative to the family's reference arm, plus which suite it came from; then the
per-suite tables. Arms that never produced a measurement show their last failure.
``--notes`` prepends a hand-written notes file; ``--loop N`` rewrites the file every N s.
"""

from __future__ import annotations

import argparse
import datetime
import re
import time
from pathlib import Path

from speed_table import markdown, table  # noqa: E402  (same directory)

REFERENCES = {"dense130M": "dense", "moe120M1200M": "moe", "moe50M500M": "moe"}
FAMILIES = tuple(REFERENCES)
# Suites measuring something other than the single-node DP-4 setup (parallelism checks,
# checkpoint save / resume, context parallel): per-suite tables only, never "latest".
OFF_SETUP = re.compile(r"^(par\d*|ck\d*)_|cp\d")


def _family(suite_dir: Path) -> str | None:
    return next((f for f in FAMILIES if suite_dir.name.endswith("_" + f)), None)


def _mtime(row: dict) -> float:
    log = row.get("log")
    return Path(log).stat().st_mtime if log and Path(log).exists() else 0.0


def collect(root: Path, skip: int) -> dict:
    """{family: {suite: rows}}"""
    out: dict = {}
    for suite in sorted(p for p in root.iterdir() if p.is_dir()):
        family = _family(suite)
        if family is None or "prof" in suite.name:  # profiled runs are slower by design
            continue
        rows = table(suite, REFERENCES[family], skip)
        if rows:
            out.setdefault(family, {})[suite.name] = rows
    return out


def latest(suites: dict, reference: str) -> list[dict]:
    """Newest measured row per arm (a failed newer run does not hide an older success,
    but is noted)."""
    best: dict = {}
    for suite, rows in suites.items():
        if OFF_SETUP.search(suite):
            continue
        for row in rows:
            row = {**row, "suite": suite}
            arm = row["arm"]
            prev = best.get(arm)
            ok, prev_ok = row["status"] == "ok", prev is not None and prev["status"] == "ok"
            if prev is None or (ok and (not prev_ok or _mtime(row) > _mtime(prev))) or (
                not ok and not prev_ok and _mtime(row) > _mtime(prev)
            ):
                best[arm] = row
    rows = list(best.values())
    ref = best.get(reference)
    for r in rows:
        r.pop("tflops_rel", None)
        r.pop("tok_rel", None)
        if ref and ref["status"] == "ok" and r["status"] == "ok":
            if r.get("tflops") and ref.get("tflops"):
                r["tflops_rel"] = r["tflops"] / ref["tflops"]
            if r.get("tok_s_gpu") and ref.get("tok_s_gpu"):
                r["tok_rel"] = r["tok_s_gpu"] / ref["tok_s_gpu"]
    order: dict = {}
    for suite, suite_rows in suites.items():  # the order of the arms in the suites
        if OFF_SETUP.search(suite):
            continue
        for row in suite_rows:
            order.setdefault(row["arm"], len(order))
    return sorted(rows, key=lambda r: (order.get(r["arm"], 1e9), r["arm"]))


def render(root: Path, skip: int, threshold: float, notes: Path | None) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [f"# Hybrid speed state\n\nupdated {now} from `{root}` (median step after "
             f"iteration {skip}; one node, 4x H100 64GB; H100 bf16 dense peak ~989 TFLOP/s)\n"]
    if notes and notes.exists():
        parts.append(notes.read_text().strip() + "\n")
    data = collect(root, skip)
    for family, suites in data.items():
        reference = REFERENCES[family]
        rows = latest(suites, reference)
        table_md = markdown(rows, reference, threshold)
        lines = table_md.splitlines()
        # add the source suite as a last column
        lines[2] = lines[2] + " suite |"
        lines[3] = lines[3] + "---|"
        for i, row in enumerate(rows):
            lines[4 + i] = lines[4 + i] + f" {row['suite']} |"
        parts.append(f"## {family}: latest per arm\n\n" + "\n".join(lines) + "\n")
    parts.append("## Per suite\n")
    for family, suites in data.items():
        for suite, rows in suites.items():
            parts.append(f"### {suite}\n\n" + markdown(rows, REFERENCES[family], threshold) + "\n")
    return "\n".join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path, help="$OUTPUT_DIR/hybrid_speed")
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--notes", type=Path)
    p.add_argument("--skip", type=int, default=30)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--loop", type=float, default=0, help="rewrite every N seconds")
    args = p.parse_args()
    while True:
        text = render(args.root, args.skip, args.threshold, args.notes)
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(text)
        tmp.replace(args.output)
        if not args.loop:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
