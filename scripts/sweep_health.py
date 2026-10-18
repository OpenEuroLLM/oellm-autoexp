#!/usr/bin/env python3
"""One-shot health check for the running multilingual main sweep.

Scans every job dir's latest log + checkpoints and prints a one-line summary
plus a VERDICT (OK / PROBLEM: ...). Designed to be polled on a timer so a
regression of the checkpoint/restart problem (or a new failure) is caught fast.

Failure modes it flags:
  - NCCL Error / OutOfMemoryError in a current log (the checkpoint-save regression
    signature, or a fresh OOM).
  - restart-from-scratch loop: a job with many attempts (slurm logs) but a low
    current iteration and no checkpoint tracker (the original bug's signature).
  - a checkpoint that stayed `common.pt`-only while its log advanced past the
    save (an incomplete/hung save).
Positive signals: complete checkpoints (>=3 files) with trackers, growing
maxiter, and whether an 8-node 0.9B save has completed (the most stressed case).
"""
import glob
import os
import re
import subprocess
import time

BASE = (
    "/e/scratch/e-sta-openeurollm/poeppel1/output/"
    "architecture_scaling_variants_multilingual_main_7to1"
)


def squeue_counts():
    try:
        out = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", "poeppel1"), "-h", "-o", "%T %j"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except Exception:
        return 0, 0
    sweep = [l for l in out.splitlines() if re.search(r"stable|firstcd|contcd|decay", l)]
    return sum("RUNNING" in l for l in sweep), sum("PENDING" in l for l in sweep)


def main():
    running, pending = squeue_counts()
    complete = incomplete_hung = nccl = oom = looping = maxiter = ckpt_09b = 0
    problems = []
    for d in sorted(glob.glob(BASE + "/*")):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        logs = sorted(glob.glob(d + "/slurm-*.log"), key=os.path.getmtime)
        if not logs:
            continue
        try:
            txt = open(logs[-1], errors="ignore").read()
        except OSError:
            continue
        its = re.findall(r"iteration +(\d+)/", txt)
        li = int(its[-1]) if its else 0
        maxiter = max(maxiter, li)
        trf = d + "/latest_checkpointed_iteration.txt"
        tr = open(trf).read().strip() if os.path.exists(trf) else ""
        for it in glob.glob(d + "/iter_*"):
            try:
                nf = len(os.listdir(it))
            except OSError:
                continue
            if nf >= 3:
                complete += 1
                if "0.9B" in name:
                    ckpt_09b = 1
        if "NCCL Error" in txt:
            nccl += 1
            problems.append(f"{name}:NCCL")
        if "OutOfMemoryError" in txt:
            oom += 1
            problems.append(f"{name}:OOM")
        # incomplete/hung save: a save was logged, no "successfully saved" after it,
        # and the newest checkpoint dir has only common.pt. Guarded against two
        # false positives: (a) a FINISHED run ("after training is done") whose
        # save is by definition resolved, and (b) an IN-FLIGHT save caught
        # mid-write — the fast 1-step firstcd saves (save_interval=1) complete in
        # well under a poll, so a dir touched in the last 180s is writing, not
        # hung. Only a stale-and-incomplete dir in an unfinished run is a real hang.
        if (
            "saving checkpoint at iteration" in txt
            and "after training is done" not in txt
            and "successfully saved checkpoint"
            not in txt.split("saving checkpoint at iteration")[-1]
        ):
            iters = sorted(glob.glob(d + "/iter_*"))
            if iters:
                try:
                    fresh = (time.time() - os.path.getmtime(iters[-1])) < 180
                    if len(os.listdir(iters[-1])) < 3 and not fresh:
                        incomplete_hung += 1
                        problems.append(f"{name}:hung-save")
                except OSError:
                    pass
        # restart-from-scratch loop
        if len(logs) > 8 and li < 300 and not tr:
            looping += 1
            problems.append(f"{name}:loop({len(logs)}att/it{li})")
    verdict = "OK" if not problems else "PROBLEM: " + " ".join(sorted(set(problems))[:8])
    print(
        f"running={running} pending={pending} complete_ckpts={complete} "
        f"maxiter={maxiter} ckpt09b={ckpt_09b} nccl={nccl} oom={oom} "
        f"hung_saves={incomplete_hung} looping={looping}"
    )
    print(f"VERDICT={verdict}")


if __name__ == "__main__":
    main()
