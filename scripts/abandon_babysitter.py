#!/usr/bin/env python3
"""Self-healing babysitter for the multilingual main sweep.

Fixes the monitor's launch-failure abandonment gap WITHOUT relaunching it:
the monitor terminal-cancels any job whose (Nth) attempt ends in SLURM state
FAILED (e.g. an empty-MASTER_ADDR torchrun exit-2 flake) and never retries it,
so the run is silently lost. This process detects such abandoned runs and
resubmits them from a robust-MASTER_ADDR copy of their sbatch, then fully owns
them (the monitor no longer tracks them, so their future TIMEOUTs are on us too).

Coexistence rules (avoid double-writing a checkpoint dir):
  - A run the MONITOR still owns is only taken over once its latest attempt is in
    SLURM state FAILED — the monitor restarts TIMEOUT itself, so those are left
    alone. FAILED is terminal to the monitor, so there is no race.
  - Once taken over, a run is in OWNED and we manage it fully: resubmit whenever
    it is not done and not in the queue (covers its own future timeouts).
  - squeue is read as a union of 3 samples to survive transient-empty glitches,
    and re-checked immediately before every sbatch.
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
OWNED_FILE = os.path.expanduser("~/work/Projects/oellm-autoexp-hybrid/babysit_owned.txt")
LOG = os.path.expanduser("~/work/Projects/oellm-autoexp-hybrid/babysit_generic.log")
ALERT = os.path.expanduser("~/work/Projects/oellm-autoexp-hybrid/HEALTH_ALERT")
POLL_S = 600

# Robust MASTER_ADDR: retry nslookup 15x, fall back to the raw hostname, never empty.
ROBUST_MA = (
    'export MASTER_ADDR=$(a=""; for i in $(seq 1 15); do '
    'a=$(nslookup $(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) '
    '| grep "Address: " | sed "s/Address: //" | tail -n 1); '
    '[ -n "$a" ] && break; sleep 2; done; '
    '[ -z "$a" ] && a=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1); '
    'echo "$a")'
)


def log(msg):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG, "a") as f:
        f.write(f"[{stamp}] {msg}\n")


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return ""


def queued_names():
    """Union of 3 squeue reads, sweep-index suffix stripped, to match dir names."""
    names = set()
    for _ in range(3):
        out = sh("squeue -u poeppel1 -h -o '%j'")
        for line in out.splitlines():
            names.add(re.sub(r"_[0-9]+$", "", line.strip()))
        time.sleep(2)
    return names


def load_owned():
    if os.path.exists(OWNED_FILE):
        return set(l.strip() for l in open(OWNED_FILE) if l.strip())
    return set()


def save_owned(owned):
    with open(OWNED_FILE, "w") as f:
        f.write("\n".join(sorted(owned)) + "\n")


def ensure_robust_script(d):
    """Write job.sbatch.robust: a copy of job.sbatch with the MASTER_ADDR line
    replaced by the robust form (the monitor may have clobbered job.sbatch back
    to the flaky version at its last submit)."""
    src, dst = f"{d}/job.sbatch", f"{d}/job.sbatch.robust"
    if not os.path.exists(src):
        return None
    lines = open(src).read().splitlines()
    out = [ROBUST_MA if l.startswith("export MASTER_ADDR=") else l for l in lines]
    with open(dst, "w") as f:
        f.write("\n".join(out) + "\n")
    return dst


def train_iters(d):
    m = re.search(r"--train-iters (\d+)", open(f"{d}/job.sbatch").read())
    return int(m.group(1)) if m else None


def latest_log(d):
    logs = sorted(glob.glob(f"{d}/slurm-*.log"), key=os.path.getmtime)
    return logs[-1] if logs else None


def is_done(d):
    ti = train_iters(d)
    trf = f"{d}/latest_checkpointed_iteration.txt"
    if os.path.exists(trf) and ti:
        try:
            if int(open(trf).read().strip()) >= ti:
                return True
        except ValueError:
            pass
    L = latest_log(d)
    if L:
        try:
            if "after training is done" in open(L, errors="ignore").read():
                return True
        except OSError:
            pass
    return False


def latest_state(d):
    L = latest_log(d)
    if not L:
        return None
    m = re.search(r"slurm-(\d+)\.log", os.path.basename(L))
    if not m:
        return None
    out = sh(f"sacct -j {m.group(1)} -X -n -o State")
    return out.split("\n")[0].strip().split()[0] if out.strip() else None


def resubmit(d):
    script = ensure_robust_script(d)
    if not script:
        return None
    # re-check queue right before submitting
    name = os.path.basename(d)
    if name in queued_names():
        return None
    out = sh(f"sbatch --parsable {script}")
    return out.strip() or None


def main():
    log("babysitter started")
    while True:
        owned = load_owned()
        q = queued_names()
        dirs = [d for d in sorted(glob.glob(BASE + "/*")) if os.path.isdir(d)]
        done_ct = sum(is_done(d) for d in dirs)
        if done_ct >= 283:  # sweep essentially complete
            log(f"sweep complete ({done_ct}/{len(dirs)}); exiting")
            with open(ALERT, "w") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M')} BABYSITTER: sweep complete {done_ct}/{len(dirs)}\n")
            break
        acted = 0
        for d in dirs:
            name = os.path.basename(d)
            if name in q or is_done(d):
                continue
            if not glob.glob(f"{d}/slurm-*.log"):
                continue  # never attempted -> legit waiting (monitor's start_condition)
            if name in owned:
                jid = resubmit(d)
                if jid:
                    log(f"OWNED resubmit {name} -> {jid}")
                    acted += 1
            else:
                st = latest_state(d)
                if st == "FAILED":
                    owned.add(name)
                    save_owned(owned)
                    jid = resubmit(d)
                    log(f"TAKEOVER (FAILED) {name} -> {jid}")
                    acted += 1
        log(f"cycle: done={done_ct}/{len(dirs)} owned={len(owned)} acted={acted}")
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
