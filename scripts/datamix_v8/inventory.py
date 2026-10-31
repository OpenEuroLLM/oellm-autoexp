"""Where is each datamix-v8 source, and how big is it? (docs/datamix_v8/datamix_v8.tsv)

    python scripts/datamix_v8/inventory.py [--out docs/datamix_v8/inventory.tsv]

Per source and cluster (MN5: the flag collection mirrors JSC's flag/ layout; JUPITER: jsc column; LUMI: lumi
column): does the tokenized Megatron data exist (.bin / .idx), how many shards, how many tokens (.bin bytes / 4,
int32). One read-only SSH call per cluster (a Python snippet over stdin; no files written remotely).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TSV = os.path.join(HERE, "..", "..", "docs", "datamix_v8", "datamix_v8.tsv")
JSC_FLAG = "/e/data1/datasets/playground/mmlaion/oellm/training/collection/flag/"
JSC_MV1 = ("/e/data1/datasets/playground/mmlaion/oellm/container/data/training/collection/flag/"
           "mixture-vitae-1.0-subsets-reasoning-instruct/")
MN5_FLAG = "/gpfs/projects/ehpc533/training-data-collection/flag/"
LUMI_VTOK = "/scratch/project_465002530/users/vitiugin/data_tok/"
HOSTS = {"mn5": "marenostrum_new", "jupiter": "jupiter", "lumi": "lumi"}

REMOTE = r'''
import json, os, sys, glob
out = {}
for key, p in json.loads(sys.stdin.read()).items():
    cands = []
    if os.path.isdir(p):
        for root, dirs, files in os.walk(p, followlinks=True):
            cands += [os.path.join(root, f) for f in files if f.endswith(".bin")]
    else:
        for c in (p, p + ".bin", p + "_text_document.bin"):
            if os.path.isfile(c) and c.endswith(".bin"):
                cands.append(c)
        cands += [c for c in glob.glob(p + "*.bin")]
    cands = sorted(set(cands))
    idx = [c for c in cands if os.path.exists(c[:-4] + ".idx")]
    out[key] = {"exists": os.path.exists(p) or bool(cands), "bins": len(cands), "with_idx": len(idx),
                "bytes": sum(os.path.getsize(c) for c in cands), "first": cands[0] if cands else ""}
print(json.dumps(out))
'''


def rows():
    return [r for r in csv.DictReader((l for l in open(TSV) if not l.startswith("#")), delimiter="\t")]


def paths(r):
    j, l = r["jsc"], r["lumi"]
    jp = (JSC_FLAG + j[5:]) if j.startswith("flag/") else (JSC_MV1 + j[7:] + "/megatron-lm") if j.startswith("mv1sub/") else ""
    mp = (MN5_FLAG + j[5:]) if j.startswith("flag/") else ""
    lp = (LUMI_VTOK + l[5:]) if l.startswith("vtok/") else (l if l.startswith("/") else "")
    return {"mn5": mp, "jupiter": jp, "lumi": lp}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "docs", "datamix_v8", "inventory.tsv"))
    a = ap.parse_args()
    rs = rows()
    res = {}
    for cl, host in HOSTS.items():
        q = {r["name"]: paths(r)[cl] for r in rs if paths(r)[cl]}
        p = subprocess.run(["ssh", "-o", "ConnectTimeout=30", host, "python3 -"], input=REMOTE.replace("sys.stdin.read()", repr(json.dumps(q))),
                           capture_output=True, text=True, timeout=1200)
        line = [l for l in p.stdout.splitlines() if l.startswith("{")]
        res[cl] = json.loads(line[-1]) if line else {}
        if not line:
            print(cl, "FAILED", p.stderr[-500:])
    with open(a.out, "w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["category", "name", "budget_B", "mn5_tokens_B", "mn5_shards", "jupiter_tokens_B", "jupiter_shards",
                    "lumi_tokens_B", "lumi_shards", "status"])
        for r in rs:
            cells, have = [], {}
            for cl in HOSTS:
                x = res.get(cl, {}).get(r["name"])
                t = x["bytes"] / 4e9 if x and x["with_idx"] else 0
                have[cl] = t
                cells += [f"{t:.1f}" if x else "-", f'{x["with_idx"]}/{x["bins"]}' if x else "-"]
            status = ("on MN5" if have["mn5"] else "JUPITER -> MN5" if have["jupiter"] else
                      "LUMI -> MN5" if have["lumi"] else "not tokenized (HF only)" if r["hf"] not in ("-", "") else "missing")
            w.writerow([r["category"], r["name"], r["budget_B"]] + cells + [status])
    print(open(a.out).read())


if __name__ == "__main__":
    main()
