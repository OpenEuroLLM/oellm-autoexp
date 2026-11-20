"""Collect the final losses on the fixed validation set (phase 1 rescoring and the scoring jobs
of the phase 2 ladders) into tables and a CSV.

Usage:
  python3 collect_rescore.py --logs $OPT_SCALE_BASE/logs [--csv rescore_phase1.csv]

Reads the Slurm logs of the scoring jobs (hyb100m_rescore_*_score{1,2,4}x[_rerun]-<jobid>.out). If a
job ran more than once, the log with the highest job id that contains a result is used.

Each score is checked against the log of the cooldown it scores: that cooldown must have
loaded exactly the checkpoint step it asked for (ckpt_step). A cooldown that branched from a
different step (Megatron prefers a newer rolling checkpoint, see ../sweep100m/README.md) is
listed under "excluded" and gets no number. A *_rerun score replaces the original one.
"""
import argparse
import csv
import glob
import os
import re

NAME = re.compile(r"hyb100m_rescore_(adam|muon|muown_classical)_lr([0-9.e-]+)_((?:beta|mom)[0-9.]+|b1-[0-9.]+_b2-[0-9.]+)_bs(\d+)_score(\d)x(_rerun)?-(\d+)\.out$")
LOSS = re.compile(r"validation loss at iteration\s+(\d+) on validation set \| lm loss value: ([0-9.E+-]+)")
LABEL = {"adam": "AdamW", "muon": "Muon", "muown_classical": "muown_classical"}
LOAD_ARG = re.compile(r"^\S*\s+load \.+ (\S+)", re.M)
CKPT_STEP = re.compile(r"^\S*\s+ckpt_step \.+ (\d+)", re.M)
LOADED = re.compile(r"successfully loaded checkpoint from \S+ .* at iteration (\d+)")


def cooldown_check(score_text, logs):
    """(ok, note) for the cooldown whose checkpoint this scoring job read."""
    m = LOAD_ARG.search(score_text)
    if not m:
        return True, "not checked: no load path in the scoring log"
    run = os.path.basename(os.path.dirname(m[1].rstrip("/")))
    cands = sorted(glob.glob(os.path.join(logs, glob.escape(run) + "-*.out")),
                   key=lambda s: int(s.rsplit("-", 1)[1][:-4]))
    for path in reversed(cands):                       # newest attempt that loaded something
        text = open(path, errors="replace").read()
        want, got = CKPT_STEP.search(text), LOADED.search(text)
        if want and got:
            if want[1] == got[1]:
                return True, ""
            return False, f"{run}: asked for step {want[1]}, loaded step {got[1]}"
    return True, f"not checked: no training log found for {run}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", required=True)
    p.add_argument("--csv", default=None, help="also write the results to this CSV file")
    a = p.parse_args()

    res, excluded, unchecked = {}, {}, set()
    # originals first, reruns last, each by job id: the last valid log of a setting wins
    for path in sorted(glob.glob(os.path.join(a.logs, "hyb100m_rescore_*_score*x*-*.out")),
                       key=lambda s: (bool(NAME.search(s)[6]), int(NAME.search(s)[7])) if NAME.search(s) else (False, 0)):
        m = NAME.search(path)
        if not m:
            continue
        text = open(path, errors="replace").read()
        hits = LOSS.findall(text)
        if hits:
            hp = m[3].replace("beta", "").replace("mom", "").replace("b1-", "").replace("_b2-", " / ")
            key = (m[1], m[5], hp, int(m[4]), float(m[2]))
            ok, note = cooldown_check(text, a.logs)
            if not ok:
                excluded[key] = note
                continue
            if note:
                unchecked.add(note)
            excluded.pop(key, None)
            res[key] = (float(hits[-1][1]), int(hits[-1][0]))
    excluded = {k: v for k, v in excluded.items() if k not in res}
    if not res:
        raise SystemExit("no rescore results found")

    lrs = sorted({k[4] for k in res})
    print(f"{len(res)} checkpoints scored (phase 1 has 162)\n")
    for budget in sorted({k[1] for k in res}):
        print(f"{budget}x budget\n")
        print("| Optimizer | beta or momentum | Batch | " + " | ".join(f"{v:g}" for v in lrs) + " |")
        print("|---|---|---|" + "---|" * len(lrs))
        for opt in LABEL:
            for hp in sorted({k[2] for k in res if k[0] == opt}):
                for bs in sorted({k[3] for k in res}):
                    row = [res.get((opt, budget, hp, bs, lr)) for lr in lrs]
                    if not any(row):
                        continue
                    best = min(r[0] for r in row if r)
                    cells = ["" if r is None else (f"**{r[0]:.4f}**" if r[0] == best else f"{r[0]:.4f}") for r in row]
                    print(f"| {LABEL[opt]} | {hp} | {bs} | " + " | ".join(cells) + " |")
        print()
    if excluded:
        print(f"excluded: {len(excluded)} scores of cooldowns that did not branch from their checkpoint step\n")
        for (opt, budget, hp, bs, lr), note in sorted(excluded.items()):
            print(f"- {LABEL[opt]} {budget}x, batch {bs}, lr {lr:g}, beta or momentum {hp}: {note}")
        print()
    for note in sorted(unchecked):
        print("note:", note)
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["optimizer", "budget", "batch_size", "lr", "beta_or_momentum", "final_step", "loss_fixed_valid"])
            for (opt, budget, hp, bs, lr), (loss, step) in sorted(res.items()):
                w.writerow([opt, f"{budget}x", bs, lr, hp, step, loss])
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
