#!/usr/bin/env python3
"""A work queue of lm-eval rows on a shared filesystem, for flag_lmeval.py --queue workers: one
engine per node that loads the model once and evaluates rows until the queue is empty (instead of
one job, one load and a guessed wall clock per group of rows).

    <queue>/pending/<order>_<item>.json   to do, the lowest order first (the longest rows first)
    <queue>/running/<order>_<item>.json   claimed: renamed out of pending/ (atomic: one worker wins)
                                          and touched by its worker while it runs
    <queue>/failed/<order>_<item>.json    gave up after ``max_attempts``

An item is a row name (``vllm/mmlu/5``) or a part of one (``vllm/mmlu/5@1of4``, lmeval_parts.py:
long rows are split so no item dominates the wall clock). Results and done markers are the
runner's own (<out>/<export>/done/<task>_<n_shot>.ok); a worker skips an item that is done. A
claim not touched for ``lease_s`` (its worker died) goes back to pending/.

    python scripts/korbi/flag_eval/flag_queue.py init --queue <dir> --done <out>/<export>/done \\
        --names vllm/hellaswag/0 vllm/mmlu/5 ... [--split-s 300 --max-parts 8]
    python scripts/korbi/flag_eval/flag_queue.py status --queue <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import statistics
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# engine seconds per row (no model load) on the fast TP1 x DP4 layout, from the done markers'
# time_s of the hd256 runs; rows without a measurement get their harness's median
ENGINE_TIMES = REPO / "config/eval_tasks/flag_evals_engine_times.json"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._@-]+", "_", name)


def plan(names: list[str], times: dict[str, float], split_s: float, max_parts: int) -> list[tuple[str, float]]:
    """(item, expected seconds), longest first; a row longer than ``split_s`` becomes
    ceil(t / split_s) parts (at most ``max_parts``)."""
    med = statistics.median(times.values()) if times else 120.0
    items = []
    for name in names:
        t = float(times.get(name, med))
        k = min(max_parts, max(1, math.ceil(t / split_s))) if split_s > 0 else 1
        if k == 1:
            items.append((name, t))
        else:
            items.extend((f"{name}@{i}of{k}", t / k) for i in range(k))
    return sorted(items, key=lambda x: -x[1])


class Queue:
    def __init__(self, root: Path, lease_s: float = 600.0, max_attempts: int = 2):
        self.root = Path(root)
        self.lease_s, self.max_attempts = lease_s, max_attempts
        for d in ("pending", "running", "failed"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def init(self, items: list[tuple[str, float]], done_dir: Path | None = None) -> int:
        """Add the items that are not done, queued, running or failed; the number added."""
        known = {json.loads(p.read_text())["item"]
                 for d in ("pending", "running", "failed") for p in (self.root / d).glob("*.json")}
        added = 0
        start = len(list((self.root / "pending").glob("*.json"))) + len(list((self.root / "running").glob("*.json")))
        for order, (item, secs) in enumerate(sorted(items, key=lambda x: -x[1]), start=start):
            if item in known or (done_dir is not None and is_done(item, done_dir)):
                continue
            path = self.root / "pending" / f"{order:05d}_{_safe(item)}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"item": item, "est_s": secs, "attempts": 0}))
            tmp.replace(path)
            added += 1
        return added

    def reclaim(self) -> int:
        """Claims not touched for lease_s back to pending (their worker is gone)."""
        n = 0
        now = time.time()
        for p in sorted((self.root / "running").glob("*.json")):
            try:
                if now - p.stat().st_mtime < self.lease_s:
                    continue
                os.rename(p, self.root / "pending" / p.name)
                n += 1
            except FileNotFoundError:
                continue  # done or reclaimed by another worker meanwhile
        return n

    def claim(self) -> tuple[Path, dict] | None:
        self.reclaim()
        for p in sorted((self.root / "pending").glob("*.json")):
            target = self.root / "running" / p.name
            try:
                os.rename(p, target)
            except FileNotFoundError:
                continue  # another worker took it
            os.utime(target)
            data = json.loads(target.read_text())
            data["worker"] = f"{socket.gethostname()}:{os.getpid()}"
            target.write_text(json.dumps(data))
            return target, data
        return None

    @staticmethod
    def touch(path: Path) -> None:
        try:
            os.utime(path)
        except FileNotFoundError:
            pass

    def complete(self, path: Path) -> None:
        path.unlink(missing_ok=True)

    def fail(self, path: Path, data: dict) -> None:
        data["attempts"] = int(data.get("attempts", 0)) + 1
        dest = "failed" if data["attempts"] >= self.max_attempts else "pending"
        target = self.root / dest / path.name
        target.write_text(json.dumps(data))
        path.unlink(missing_ok=True)

    def status(self) -> dict[str, int]:
        return {d: len(list((self.root / d).glob("*.json"))) for d in ("pending", "running", "failed")}


def is_done(item: str, done_dir: Path) -> bool:
    """The runner's markers: the row's (also covers its parts) or the part's own."""
    from scripts.korbi.flag_eval import lmeval_parts

    base, k, n = lmeval_parts.parse(item)
    _, task, n_shot = base.split("/", 2)
    if (Path(done_dir) / f"{task}_{n_shot}.ok").exists():
        return True
    return k is not None and (Path(done_dir) / f"{lmeval_parts.part_key(task, int(n_shot), k, n)}.ok").exists()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("init")
    pi.add_argument("--queue", type=Path, required=True)
    pi.add_argument("--done", type=Path, help="the runner's done dir: done rows are not queued")
    pi.add_argument("--names", nargs="+", required=True)
    pi.add_argument("--times", type=Path, default=ENGINE_TIMES)
    pi.add_argument("--split-s", type=float, default=300.0, help="split rows longer than this (0: never)")
    pi.add_argument("--max-parts", type=int, default=8)
    ps = sub.add_parser("status")
    ps.add_argument("--queue", type=Path, required=True)
    a = p.parse_args(argv)
    if a.cmd == "init":
        times = json.loads(a.times.read_text()) if a.times.exists() else {}
        items = plan(a.names, times, a.split_s, a.max_parts)
        q = Queue(a.queue)
        added = q.init(items, a.done)
        print(f"queued {added} of {len(items)} items ({len(a.names)} rows); {q.status()}")
    else:
        print(json.dumps(Queue(a.queue).status()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
