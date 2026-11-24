#!/usr/bin/env python3
"""Move the unevaluated midtraining evals from the o134c pool into p264's buffer nodes.

Why: the evals are 1-node jobs, but they sit in a pool whose only host is a parked 134-node
allocation. The main run's host010 (p264) has 4 free buffer nodes, so the evals belong there at a
low priority: they use idle nodes and are preempted if the main run needs them.

How: a queued pool run is defined entirely by its job record in the pool's monitor session
(``<session>/<job_id>.job.json``), and the monitor submits any record with ``submitted=False``.
The record carries the resolved config, so nothing has to be re-planned: copy the record, point its
``definition.pool_dir`` at p264, set ``metadata.pool_priority`` below the main run, clear the runtime
and write it into the p264 session under a new job id.

Only the evals that have no completed log are moved. A run that needs more nodes than the buffer
offers is reported and left where it is.

Usage:
  python move_evals_to_p264.py --dry-run     # print what would be written, change nothing
  python move_evals_to_p264.py --apply       # write the records
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import sys
from pathlib import Path

R = Path("/gpfs/projects/ehpc533/users/korbip/hybrid_v019")
POOLS = Path("/gpfs/scratch/ehpc533/hybrid_nucleus/pools")
SRC_POOL = POOLS / "mn5_pool_o134c"
DST_POOL = POOLS / "mn5_pool_p264"
SRC_SESSION = R / "pool_state" / "pool_mn5-pool-o134c"
DST_SESSION = R / "pool_state" / "pool_mn5-pool_p264"
BUFFER_NODES = 4          # host010: 260 allocated, 256 taken by the main run
PRIORITY = 30             # well below the main run's 150
PREFIX = "ev_ambitious_v4_2ph_hd256_fp8_"


def queued_runs() -> dict[str, dict]:
    """name -> run.json of every unfinished run in the source pool."""
    out = {}
    for d in glob.glob(str(SRC_POOL / "runs" / "*")):
        if os.path.exists(os.path.join(d, "rc")):
            continue
        try:
            m = json.load(open(os.path.join(d, "run.json")))
        except Exception:
            continue
        out[m.get("name") or os.path.basename(d)] = m
    return out


def completed_logs() -> set[str]:
    """Names that already have a finished eval log in the source pool."""
    done = set()
    for f in glob.glob(str(SRC_POOL / "slurm-*.log")):
        m = re.search(r"-([a-z0-9_]+)\.log$", f)
        if m:
            done.add(m.group(1))
    return done


def record_for(name: str) -> tuple[Path, dict] | None:
    """The newest job record in the source session whose definition name is ``name``."""
    best = None
    for p in SRC_SESSION.glob("*.job.json"):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        defn = d.get("definition") or {}
        if defn.get("name") == name:
            mt = p.stat().st_mtime
            if best is None or mt > best[0]:
                best = (mt, p, d)
    return (best[1], best[2]) if best else None


def new_job_id(name: str) -> str:
    import hashlib
    return name + "_" + hashlib.sha1((name + "-p264").encode()).hexdigest()[:6]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the records (default: dry run)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan only")
    args = ap.parse_args(argv)
    apply_ = args.apply and not args.dry_run

    runs = queued_runs()
    done = completed_logs()
    todo = [(n, m) for n, m in sorted(runs.items()) if n.startswith(PREFIX) and n not in done]

    print("queued in o134c: %d   already evaluated: %d   to move: %d"
          % (len(runs), len(runs) - len(todo), len(todo)))
    moved, skipped = [], []
    for name, meta in todo:
        nodes = int(meta.get("nodes") or 1)
        if nodes > BUFFER_NODES:
            print("  SKIP %s: needs %d nodes, the buffer has %d" % (name, nodes, BUFFER_NODES))
            skipped.append(name)
            continue
        found = record_for(name)
        if found is None:
            print("  SKIP %s: no job record in %s" % (name, SRC_SESSION))
            skipped.append(name)
            continue
        path, payload = found
        job_id = new_job_id(name)
        dst = DST_SESSION / ("%s.job.json" % job_id)
        if dst.exists():
            print("  SKIP %s: %s already exists" % (name, dst.name))
            skipped.append(name)
            continue
        rec = copy.deepcopy(payload)
        defn = rec.setdefault("definition", {})
        defn["pool_dir"] = str(DST_POOL)
        meta_d = defn.setdefault("metadata", {}) or {}
        meta_d["pool_priority"] = PRIORITY
        defn["metadata"] = meta_d
        rec["job_id"] = job_id
        rec["runtime"] = {
            "class_name": "JobRuntime",
            "submitted": False,
            "attempts": 0,
            "runtime_job_id": None,
            "start_ts": None,
            "end_ts": None,
            "log_cursor": 0,
            "condition_state": {},
            "action_state": {},
            "events": {},
            "last_status": None,
            "final_state": None,
            "deferred_restart": {},
        }
        print("  %s %-58s nodes=%d  <- %s  -> %s"
              % ("WRITE" if apply_ else "would write", name, nodes, path.name, dst.name))
        if apply_:
            DST_SESSION.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
            os.replace(tmp, dst)
        moved.append(name)

    print("moved: %d   skipped: %d" % (len(moved), len(skipped)))
    for n in skipped:
        print("  left behind:", n)
    if not apply_:
        print("dry run: nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
