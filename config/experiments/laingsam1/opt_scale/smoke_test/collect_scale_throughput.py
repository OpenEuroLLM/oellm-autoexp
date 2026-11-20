"""Table of the results of smoke tests 17 and 18 (17_scale_throughput.yaml, 18_tput_250m_500m.yaml),
from the Slurm logs. Test 17 has no batch size in its job names; its batch is 256.

Usage (standard library only):
  python3 collect_scale_throughput.py --logs $OPT_SCALE_BASE/logs [--csv out.csv]

Per job: median step time and tokens/s/GPU over the measured steps (the first --skip steps are
dropped as warm-up), peak GPU memory, the speed per GPU relative to the 1-node job with the same
size, expert placement, batch size and optimizer, GPU-h per billion tokens, and GPU-h for one ladder (1x, 2x
and 4x budgets from one stable run = about 92 x active parameters in tokens, as in sweep100m).
A job that ran out of memory or has too few steps is listed with its state instead of numbers.
"""
import argparse
import csv
import glob
import os
import re
import statistics

NAME = re.compile(r"scale_tput_(\d+m)(?:_bs(\d+))?_N(\d+)_(ep\d+(?:_[a-z0-9]+)*?)_(adam|muon|muown_classical)-(\d+)\.out$")
STEP = re.compile(r"iteration\s+(\d+)/\s*\d+ \|.*?elapsed time per iteration \(ms\): ([0-9.]+) \| mem usages: ([0-9.]+)"
                  r".*?throughput per GPU \(TFLOP/s/GPU\): ([0-9.]+)"
                  r".*?Tokens per second per GPU \(Tok/s/GPU\): ([0-9.]+)")
PEAK = 989   # TFLOP/s, dense bf16 peak of the GH200's GPU; mfu = Megatron's printed TFLOP/s/GPU / PEAK
MAXMEM = re.compile(r"max allocated: ([0-9.]+)")
MBS = re.compile(r"  micro_batch_size \.+ (\d+)")
ACTIVE = {"100m": 101e6, "250m": 246e6, "500m": 486e6, "850m": 845e6, "1800m": 1.79e9}   # as in the yaml header
LADDER = 92   # tokens of one ladder / active parameters


def read(path, skip):
    t = open(path, errors="replace").read()
    steps = [(int(a), float(b), float(c), float(e), float(d)) for a, b, c, d, e in STEP.findall(t)]
    used = [s for s in steps if s[0] > skip]
    if re.search(r"out of memory|OutOfMemoryError", t, re.I):
        state = "out of memory"
    elif len(used) < 5:
        state = f"only {len(steps)} steps" if steps else ("failed" if "Traceback" in t or "Exited with exit code" in t else "no steps yet")
    else:
        state = "ok"
    mb = MBS.search(t)
    out = dict(state=state, n=len(used), mbs=mb[1] if mb else "")
    if state == "ok":
        out.update(ms=statistics.median(s[1] for s in used), tok=statistics.median(s[3] for s in used),
                   mem=max(s[2] for s in used), tflop=statistics.median(s[4] for s in used))
        mm = MAXMEM.findall(t)
        out["gb"] = max(map(float, mm)) / 1024 if mm else None
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", required=True, help="directory with the Slurm .out files")
    p.add_argument("--skip", type=int, default=5, help="warm-up steps to drop")
    p.add_argument("--csv", help="also write the table to this file")
    a = p.parse_args()

    rows = {}
    for path in sorted(glob.glob(os.path.join(a.logs, "scale_tput_*.out")), key=lambda q: int(q.rsplit("-", 1)[1][:-4])):
        m = NAME.search(path)
        if m:   # a later job id of the same setting replaces an earlier one
            rows[(m[1], m[5], m[4], int(m[2] or 256), int(m[3]))] = dict(read(path, a.skip), job=m[6])
    if not rows:
        raise SystemExit("no scale_tput_*.out logs found")

    table = []
    order = list(ACTIVE)
    for key in sorted(rows, key=lambda k: (order.index(k[0]) if k[0] in order else 99,) + k[1:]):
        size, opt, ep, batch, nodes = key
        r = rows[key]
        row = dict(size=size, optimizer=opt, placement=ep, batch=batch, nodes=nodes, micro_batch=r["mbs"], job=r["job"], state=r["state"])
        if r["state"] == "ok":
            base = rows.get((size, opt, ep, batch, 1))
            per_bt = 1e9 / r["tok"] / 3600
            row.update(s_per_step=round(r["ms"] / 1000, 2), tok_s_gpu=round(r["tok"]), tflop_gpu=round(r["tflop"], 1),
                       mfu_pct=round(100 * r["tflop"] / PEAK, 1), mem_frac=r["mem"],
                       max_alloc_gb=round(r["gb"], 1) if r["gb"] else "",
                       vs_1_node=round(r["tok"] / base["tok"], 2) if base and base["state"] == "ok" else "",
                       gpuh_per_bt=round(per_bt, 1),
                       gpuh_ladder=round(per_bt * LADDER * ACTIVE[size] / 1e9) if size in ACTIVE else "",
                       hours_ladder=round(per_bt * LADDER * ACTIVE[size] / 1e9 / (4 * nodes), 1) if size in ACTIVE else "")
        table.append(row)

    cols = ["size", "optimizer", "placement", "batch", "nodes", "micro_batch", "state", "s_per_step", "tok_s_gpu", "tflop_gpu", "mfu_pct", "vs_1_node", "mem_frac",
            "max_alloc_gb", "gpuh_per_bt", "gpuh_ladder", "hours_ladder", "job"]
    width = {c: max(len(c), *(len(str(r.get(c, ""))) for r in table)) for c in cols}
    print("  ".join(c.ljust(width[c]) for c in cols))
    last = None
    for r in table:
        if last and (r["size"], r["optimizer"]) != last:
            print()
        last = (r["size"], r["optimizer"])
        print("  ".join(str(r.get(c, "")).ljust(width[c]) for c in cols))
    print("\ntflop_gpu: Megatron's own figure from the log (includes the output layer and attention); mfu_pct = tflop_gpu / 989."
          "\nvs_1_node: tokens/s/GPU relative to the 1-node job of the same size, placement, batch and optimizer (1.0 = no loss)."
          "\ngpuh_ladder / hours_ladder: one ladder (1x, 2x, 4x) at this speed; hours on this many nodes, without queue time.")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows({c: r.get(c, "") for c in cols} for r in table)
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
