#!/usr/bin/env python3
"""Reconcile the multilingual main sweep against its expected 285 runs.

Expected = 5 archs x (7 stables + 25 budget-points x {firstcd, contcd}) = 285.
For each expected run, classify:
  DONE      - "after training is done" logged AND checkpoint at the target iter
  ACTIVE    - currently in the SLURM queue (running/pending)
  MISSING   - attempted (has a log) but not done and not queued  -> SILENT FAILURE
  WAITING   - not started, prerequisite checkpoint not ready yet  -> legitimate
  NOSTART   - not started but prerequisite IS ready               -> SILENT (never fired)
Also flags cooldowns that "finished" at the firstcd seed (start_iter+1) instead
of the full budget iteration -> the decay never actually ran (silent).
"""
import glob
import math
import os
import re
import subprocess

BASE = ("/e/scratch/e-sta-openeurollm/poeppel1/output/"
        "architecture_scaling_variants_multilingual_main_7to1")
ARCHS = ["fullattn", "swa7_rope100k", "gdn7_nope", "mlstm7_nope", "mamba2_nope"]
# (size, gbs, lr, [budgets_BT])
LADDERS = [
    ("0.1B", 128, "0.002", [20, 30, 50]),
    ("0.1B", 256, "0.002", [80, 200, 300]),
    ("0.2B", 128, "0.001", [6, 12, 20, 30]),
    ("0.2B", 512, "0.002", [50, 80, 120, 200, 300]),
    ("0.4B", 256, "0.002", [12, 20, 30, 50]),
    ("0.9B", 512, "0.0005", [30, 50, 80, 120]),
    ("0.9B", 1024, "0.001", [200, 300]),
]
SEQ, DECAY = 4096, 0.2


def full_iters(bt, gbs):
    return math.ceil(bt * 1e9 / (SEQ * gbs))


def branch_iter(bt, gbs):
    return int(full_iters(bt, gbs) * (1 - DECAY))


def squeue_names():
    out = subprocess.run(["squeue", "-u", "poeppel1", "-h", "-o", "%j"],
                         capture_output=True, text=True).stdout
    return set(re.sub(r"_[0-9]+$", "", l.strip()) for l in out.splitlines())


def latest_log(d):
    logs = [p for p in glob.glob(d + "/slurm-*.log")]
    logs.sort(key=lambda p: os.path.getmtime(p))
    return logs[-1] if logs else None


def log_done(d):
    L = latest_log(d)
    if not L:
        return False, False
    try:
        return ("after training is done" in open(L, errors="ignore").read()), True
    except OSError:
        return False, True


def max_ckpt_iter(d):
    its = [int(os.path.basename(p).split("_")[1]) for p in glob.glob(d + "/iter_*")
           if re.match(r"iter_\d+$", os.path.basename(p))]
    return max(its) if its else None


def main():
    q = squeue_names()
    stables_target = {}  # (arch,size,gbs,lr) -> max branch iter (stable trains to 0.8*Dmax)
    for arch in ARCHS:
        for (size, gbs, lr, buds) in LADDERS:
            stables_target[(arch, size, gbs, lr)] = branch_iter(max(buds), gbs)

    cats = {"DONE": [], "ACTIVE": [], "MISSING": [], "WAITING": [], "NOSTART": [],
            "SEED_ONLY": []}
    n_expected = 0

    for arch in ARCHS:
        for (size, gbs, lr, buds) in LADDERS:
            base = f"qwen3_{arch}_{size}_ne_lr{lr}_gbsz{gbs}"
            # ---- stable ----
            n_expected += 1
            sd = f"{BASE}/{base}_stable"
            done, haslog = log_done(sd)
            tgt = stables_target[(arch, size, gbs, lr)]
            ck = max_ckpt_iter(sd)
            name = f"{base}_stable"
            if name in q:
                cats["ACTIVE"].append(name)
            elif done or (ck is not None and ck >= tgt):
                cats["DONE"].append(name)
            elif haslog:
                cats["MISSING"].append(f"{name} (ck={ck}/{tgt})")
            else:
                cats["NOSTART"].append(name)
            stable_done = done or (ck is not None and ck >= tgt)

            for bt in buds:
                fi, bi = full_iters(bt, gbs), branch_iter(bt, gbs)
                # ---- firstcd (seeds the decay dir at bi+1) ----
                n_expected += 1
                fname = f"{base}_firstcd_decay{bt}BT"
                fd = f"{BASE}/{fname}"
                fdone, fhas = log_done(fd)
                fck = max_ckpt_iter(fd)   # this dir also holds contcd's saves
                if fname in q:
                    cats["ACTIVE"].append(fname)
                elif fdone or (fck is not None and fck >= bi):
                    cats["DONE"].append(fname)
                elif fhas:
                    cats["MISSING"].append(f"{fname} (silent-fail)")
                elif stable_done:
                    cats["NOSTART"].append(f"{fname} (stable done, never fired)")
                else:
                    cats["WAITING"].append(fname)

                # ---- contcd (decays in the firstcd dir to full budget iter fi) ----
                n_expected += 1
                cname = f"{base}_contcd_decay{bt}BT"
                cd = f"{BASE}/{cname}"
                cdone, chas = log_done(cd)
                # contcd checkpoints land in the firstcd dir; completion => fck>=fi
                if cname in q:
                    cats["ACTIVE"].append(cname)
                elif fck is not None and fck >= fi:
                    cats["DONE"].append(cname)
                elif cdone and fck is not None and fck <= bi + 2:
                    # log says done but the dir only has the seed -> decay never ran
                    cats["SEED_ONLY"].append(f"{cname} (ck={fck}, want {fi})")
                elif chas:
                    cats["MISSING"].append(f"{cname} (ck={fck}/{fi}, silent-fail)")
                elif stable_done:
                    cats["NOSTART"].append(f"{cname} (firstcd seed ready? never fired)")
                else:
                    cats["WAITING"].append(cname)

    print(f"expected runs: {n_expected}")
    for k in ["DONE", "ACTIVE", "WAITING", "MISSING", "NOSTART", "SEED_ONLY"]:
        print(f"  {k:9s}: {len(cats[k])}")
    total = sum(len(v) for v in cats.values())
    print(f"  {'TOTAL':9s}: {total}  ({'OK' if total == n_expected else 'MISMATCH'})")
    for k in ["MISSING", "NOSTART", "SEED_ONLY"]:
        if cats[k]:
            print(f"\n=== {k} ({len(cats[k])}) ===")
            for x in sorted(cats[k]):
                print(f"  {x}")


if __name__ == "__main__":
    main()
