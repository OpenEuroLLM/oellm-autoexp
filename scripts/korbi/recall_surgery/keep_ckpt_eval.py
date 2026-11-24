#!/usr/bin/env python3
"""Archive every N-step persistent checkpoint into <run>/ckpt_eval as a hard-link farm.

Why this exists: the training loop's retention deletes the previous persistent save at each
persistent save. Observed on the v4 continuation (run .1615, 2026-11-18):

  * at the 39000 save, iter_0038000 was removed;
  * at the 39218 exit-signal save, iter_0039000 was removed -- a 10000 multiple, which
    --save-retain-interval is supposed to keep. Two code copies in this tree contain that
    guard (`prev_iteration % save_retain_interval != 0`), so the running fork's policy is
    not what those copies say; the operative rule is the observed one: the previous
    persistent save goes away.

So a 1000-step archive cannot be had from retention. It has to be a mirror, which is the
repo's existing convention: broad_readout.py hard-links a save into ckpt_eval/iter_N before
it can be pruned. Hard links cost no space (same GPFS filesystem), and pruning the source
only decrements the link count -- the data stays.

The trigger is the persistent tracker, not the log: it is written after the save completes
and before the previous save is deleted, so it is authoritative and survives run restarts.

  python3 scripts/korbi/recall_surgery/keep_ckpt_eval.py --run-dir <run> [--every 1000]
  python3 scripts/korbi/recall_surgery/keep_ckpt_eval.py --run-dir <run> --once
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time


def tracker(run_dir: str) -> int | None:
    try:
        with open(os.path.join(run_dir, "ckpt", "latest_checkpointed_iteration.txt")) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def mirror(run_dir: str, iteration: int) -> str:
    """Hard-link ckpt/iter_N to ckpt_eval/iter_N. Returns a one-line outcome."""
    name = f"iter_{iteration:07d}"
    dst = os.path.join(run_dir, "ckpt_eval", name)
    if os.path.isdir(dst):
        return f"{name}: already in ckpt_eval"
    src = os.path.join(run_dir, "ckpt", name)
    if not os.path.isdir(src):
        # The persistent copy is gone (pruned by the next save); the rolling copy is the
        # only remaining source, and it is itself short-lived, so use it if it is there.
        rolling = os.path.join(run_dir, "ckpt_rolling", name)
        if not os.path.isdir(rolling):
            return f"{name}: not found in ckpt/ or ckpt_rolling/ (lost to retention)"
        src = rolling
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # cp -al: hard links, no data copied; -n is not needed because dst is absent.
    res = subprocess.run(["cp", "-al", src, dst], capture_output=True, text=True)
    if res.returncode != 0:
        return f"{name}: FAILED cp -al: {res.stderr.strip()[:200]}"
    want = len(os.listdir(src))
    got = len(os.listdir(dst))
    if want != got:
        return f"{name}: FAILED count mismatch src={want} dst={got}"
    sample = next((f for f in os.listdir(dst) if f.endswith(".distcp")), None)
    links = os.stat(os.path.join(dst, sample)).st_nlink if sample else -1
    return f"{name}: mirrored ({got} file(s), link count {links})"


def trim(run_dir: str, keep: int, every: int) -> int:
    """Drop the oldest *own* mirrors beyond `keep`.

    ckpt_eval is a shared convention: the export/eval pipelines of other arms hard-link
    their sources there too (ten such entries appeared in v4b's ckpt_eval on 2026-11-18).
    Trimming by age alone would delete someone else's archive, so only iterations that are
    multiples of `every` -- the ones this tool mirrors -- are candidates.
    """
    ev = os.path.join(run_dir, "ckpt_eval")
    dirs = sorted(
        d
        for d in os.listdir(ev)
        if d.startswith("iter_")
        and os.path.isdir(os.path.join(ev, d))
        and d.split("_")[-1].isdigit()
        and int(d.split("_")[-1]) % every == 0
    )
    removed = 0
    for name in dirs[:-keep] if keep > 0 else []:
        # Name what goes: a silent count cannot be audited afterwards (2026-11-23: fourteen
        # "trimmed 1" lines in an hour with no record of which iterations were destroyed).
        print(f"  trim: removing {name}", flush=True)
        shutil.rmtree(os.path.join(ev, name), ignore_errors=True)
        removed += 1
    return removed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="e.g. .../production/v4_cont/runs/main")
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument(
        "--from",
        dest="from_iteration",
        type=int,
        default=0,
        help="first iteration to consider (skips the history a continued run inherits)",
    )
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--budget", type=float, default=0.0, help="seconds; 0 = run until stopped")
    ap.add_argument("--once", action="store_true", help="one pass over the current tracker, then exit")
    ap.add_argument(
        "--keep",
        type=int,
        default=0,
        help="keep only the newest N mirrors (0 = unlimited). Each archived step is a real "
        "1.1 TB on disk once the source is pruned, so an uncapped archive needs a plan.",
    )
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(os.path.join(run_dir, "ckpt_eval"), exist_ok=True)
    # A continued run inherits a long history: start at the first multiple of `every` at or
    # after --from, so the first pass does not walk (and miss) every old iteration.
    first = -(-args.from_iteration // args.every) * args.every
    seen: set[int] = {i for i in range(args.every, first, args.every)}
    t0 = time.time()
    print(f"keep_ckpt_eval: {run_dir} every {args.every} step(s), interval {args.interval:.0f}s")
    while True:
        now = tracker(run_dir)
        if now is not None:
            for iteration in range(args.every, now + 1, args.every):
                if iteration in seen:
                    continue
                seen.add(iteration)
                line = mirror(run_dir, iteration)
                print(f"[{time.strftime('%H:%M:%S')}] {line}", flush=True)
                if "FAILED" in line:
                    return 4
        else:
            print(f"[{time.strftime('%H:%M:%S')}] no tracker yet in {run_dir}/ckpt", flush=True)
        if args.keep > 0:
            trimmed = trim(run_dir, args.keep, args.every)
            if trimmed:
                print(f"[{time.strftime('%H:%M:%S')}] trimmed {trimmed} old mirror(s) (keep {args.keep})", flush=True)
            elif os.environ.get("KEEP_CKPT_EVAL_DEBUG"):
                own = [d for d in sorted(os.listdir(os.path.join(run_dir, "ckpt_eval")))
                       if d.startswith("iter_") and d.split("_")[-1].isdigit()
                       and int(d.split("_")[-1]) % args.every == 0]
                print(f"[{time.strftime('%H:%M:%S')}] no trim: {len(own)} own mirror(s) <= keep {args.keep}", flush=True)
        if args.once:
            return 0
        if args.budget and time.time() - t0 >= args.budget:
            print(f"budget spent; tracker {now}", flush=True)
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
