#!/usr/bin/env python3
"""Compare the ambitious resume test (hybrid_ambitious_resume_test_mn5.yaml).

1. State: Megatron's MEGATRON_DEBUG_OPT_STATE_FILE dumps of ref at its step-20 save against
   cont's / grow's dumps after loading that checkpoint. Per parameter (model file, by name) and per
   optimizer entry (chain.group.index), the sum of squares is summed over ranks, so layouts with a
   different DP size compare; the same layout (cont) also compares rank by rank, position-weighted
   sums included (catches permuted rows / swapped halves). Under Megatron-FSDP the optimizer
   dump is the verdict: its param= is the fp32 main shard. The model file reads the bf16 module
   parameters, views into all-gather buffers that are freed / stale between steps (after a load
   some read 0), so it is printed for information only.
2. Training: lm loss / grad norm / load balancing per step from the slurm logs, cont / grow / ref2
   against ref.

    python scripts/korbi/production/compare_resume_test.py /gpfs/scratch/ehpc533/hybrid_nucleus/resume_tests/amb_rs
"""

import glob
import math
import os
import re
import sys
from collections import defaultdict


def parse_value(v):
    if v in ("0", "freed"):
        return (0.0, 0.0) if v == "0" else None
    if "/" in v:
        a, b = v.split("/")
        return float(a), float(b)
    try:
        return float(v)
    except ValueError:
        return v


def read_dump(path):
    """{entry: {field: value}} of one rank's file (model: field 'w')."""
    out = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"(\S+) (\([^)]*\)) (.*)$", line.strip())
            if not m:
                continue
            name, shape, rest = m.groups()
            fields = {"shape": shape}
            for tok in rest.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    fields[k] = parse_value(v) if k != "link" else v
                else:
                    fields["w"] = parse_value(tok)
            out[name] = fields
    return out


def dumps(root, arm, tag, kind):
    pat = f"{root}/{arm}/opt_state/dump.{tag}{'.model' if kind == 'model' else ''}.rank*.txt"
    files = sorted(glob.glob(pat), key=lambda p: int(re.search(r"rank(\d+)", p).group(1)))
    return {int(re.search(r"rank(\d+)", p).group(1)): read_dump(p) for p in files}


def summed(per_rank, skip=("grad", "shape", "link")):
    """{(entry, field): summed sum of squares} over ranks; scalars: the set of values."""
    tot = defaultdict(float)
    scal = defaultdict(set)
    for d in per_rank.values():
        for name, fields in d.items():
            for k, v in fields.items():
                if k in skip or v is None:
                    continue
                if isinstance(v, tuple):
                    tot[(name, k)] += v[0]
                elif isinstance(v, float):
                    scal[(name, k)].add(v)
    return tot, scal


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


def compare_state(root, arm, kind):
    ref = dumps(root, "ref", "at_save_20", kind)
    new = dumps(root, arm, "after_load", kind)
    if not ref or not new:
        print(f"  [{kind}] missing dumps: ref {len(ref)} ranks, {arm} {len(new)} ranks")
        return False
    rt, rs = summed(ref)
    nt, ns = summed(new)
    ok = True
    missing = sorted(set(rt) ^ set(nt))
    if missing:
        ok = False
        print(f"  [{kind}] {len(missing)} entries in only one side, e.g. {missing[:5]}")
    worst = sorted(((rel(rt[k], nt[k]), k) for k in set(rt) & set(nt)), reverse=True)
    bad = [(r, k) for r, k in worst if r > 1e-6]
    zero_loaded = [k for k in set(rt) & set(nt) if rt[k] > 0 and nt[k] == 0]
    print(f"  [{kind}] {len(worst)} summed entries ({len(ref)} -> {len(new)} ranks): "
          f"max rel diff {worst[0][0]:.2e} ({worst[0][1]}), > 1e-6: {len(bad)}, "
          f"nonzero saved / zero loaded: {len(zero_loaded)}")
    for r, k in bad[:10]:
        print(f"      {r:.3e}  {k}  saved {rt[k]:.6e}  loaded {nt[k]:.6e}")
    for k in sorted(set(rs) | set(ns)):
        if rs.get(k) != ns.get(k):
            ok = False
            print(f"      scalar {k}: saved {sorted(rs.get(k, []))} loaded {sorted(ns.get(k, []))}")
    ok = ok and not bad and not zero_loaded
    if len(ref) == len(new):  # same layout: rank by rank, position-weighted sums too
        nbad = 0
        for r in ref:
            for name, fields in ref[r].items():
                for k, v in fields.items():
                    if k in ("grad", "shape", "link") or not isinstance(v, tuple):
                        continue
                    w = new[r].get(name, {}).get(k)
                    if not isinstance(w, tuple) or rel(v[0], w[0]) > 1e-6 or rel(v[1], w[1]) > 1e-6:
                        nbad += 1
                        if nbad <= 5:
                            print(f"      rank {r} {name} {k}: saved {v} loaded {w}")
        print(f"  [{kind}] rank-by-rank (sum sq + position-weighted): {nbad} mismatches")
        ok = ok and nbad == 0
    return ok


LINE = re.compile(r"iteration +(\d+)/ *\d+ \|.*")


def read_log(root, arm):
    steps = {}
    logs = sorted(glob.glob(f"{root}/{arm}/slurm-*.log"), key=os.path.getmtime)
    for p in logs[-1:]:  # the newest job of the arm
        with open(p, errors="replace") as f:
            for line in f:
                m = LINE.search(line)
                if not m:
                    continue
                vals = {}
                for key in ("lm loss", "grad norm", "load_balancing_loss"):
                    mm = re.search(rf"\| {key}: ([0-9.eE+-]+|nan|inf)", line)
                    if mm:
                        vals[key] = float(mm.group(1))
                steps[int(m.group(1))] = vals
    # log_interval 1: the second step after a resume prints the mean of the first two (memory:
    # resume-first-log-averages-two-steps); undo it for the arms that resumed
    first = min(steps) if steps else None
    if first and first > 1 and first + 1 in steps:
        for key in ("lm loss", "load_balancing_loss"):
            if key in steps[first] and key in steps[first + 1]:
                steps[first + 1][key] = 2 * steps[first + 1][key] - steps[first][key]
    return steps


def compare_losses(root):
    ref = read_log(root, "ref")
    print(f"\nper-step metrics vs ref ({len(ref)} steps logged by ref)")
    for arm in ("ref2", "cont", "grow"):
        log = read_log(root, arm)
        common = sorted(set(ref) & set(log))
        if not common:
            print(f"  {arm}: no common steps")
            continue
        for key in ("lm loss", "grad norm", "load_balancing_loss"):
            ds = [(s, log[s][key] - ref[s][key]) for s in common if key in log[s] and key in ref[s]]
            if not ds:
                continue
            post = [d for s, d in ds if s > 20]
            pre = [d for s, d in ds if s <= 20]
            fmt = lambda xs: (f"max|d| {max(abs(x) for x in xs):.3e} rms {math.sqrt(sum(x * x for x in xs) / len(xs)):.3e}"
                              if xs else "-")
            print(f"  {arm:5s} {key:20s} steps {common[0]}-{common[-1]}  <=20: {fmt(pre)}  >20: {fmt(post)}")
        if arm != "ref2":
            print("        step  lm loss (ref / arm)       grad norm (ref / arm)")
            for s in [s for s in common if s > 20]:
                print(f"        {s:4d}  {ref[s].get('lm loss', float('nan')):.6f} / {log[s].get('lm loss', float('nan')):.6f}"
                      f"   {ref[s].get('grad norm', float('nan')):.4f} / {log[s].get('grad norm', float('nan')):.4f}")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/gpfs/scratch/ehpc533/hybrid_nucleus/resume_tests/amb_rs"
    verdict = {}
    for arm in ("cont", "grow"):
        print(f"== state: ref step-20 save vs {arm} after load")
        compare_state(root, arm, "model")
        verdict[arm] = compare_state(root, arm, "opt")
    compare_losses(root)
    print("\nstate verdict:", {a: ("exact" if v else "MISMATCH") for a, v in verdict.items()})


if __name__ == "__main__":
    main()
