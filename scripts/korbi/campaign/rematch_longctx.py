"""Long-context results of the GQA rematch (ds5lcresgqa_s2 and its log-n twin): per arm the
long-document loss by position, 64-fact retrieval by length, the 16-fact retrieval and passkey at
64k, and variable tracking -- plain, with eval-time log-n in brackets. 50 questions per retrieval
length (standard error ~0.07).

    python3 scripts/korbi/campaign/rematch_longctx.py [<campaign_3b dir>]
"""

import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/gpfs/scratch/ehpc533/hybrid_nucleus/campaign_3b")
ARMS = ["plain", "mhc4", "mhc4_lr0125", "ldm", "ldm_lr2"]
LENGTHS = ["4k", "8k", "16k", "32k", "64k"]


def load(suite, arm):
    f = ROOT / suite / arm / "ckpt" / "downstream.json"
    return json.loads(f.read_text()) if f.exists() else None


def acc(d, key):
    return d[key]["acc"] if d and key in d else None


def cell(a, b, fmt="{:.2f}"):
    if a is None:
        return ""
    return fmt.format(a) + (f" ({fmt.format(b)})" if b is not None else "")


def main():
    print("| arm | NLL 4-8k | NLL 16-32k | NLL 32-64k | kvret64 " + " / ".join(LENGTHS)
          + " | kvret16 @64k | passkey 64k | vt " + " / ".join(LENGTHS) + " |")
    print("|---|---:|---:|---:|---|---:|---:|---|")
    for arm in ARMS:
        p, g = load("ds5lcresgqa_s2", arm), load("ds5lcresgqalogn_s2", arm)
        if p is None and g is None:
            print(f"| {arm} | pending |||||||")
            continue
        lp = (p or {}).get("longppl_64k", {})
        lg = (g or {}).get("longppl_64k", {})
        nll = [cell(lp.get(f"nll_{r}"), lg.get(f"nll_{r}"), "{:.3f}")
               for r in ("4096-8192", "16384-32768", "32768-65536")]
        kv = " / ".join(cell(acc(p, f"kvret64_{n}"), acc(g, f"kvret64_{n}")) for n in LENGTHS)
        vt = " / ".join(cell(acc(p, f"vt_{n}"), acc(g, f"vt_{n}")) for n in LENGTHS)
        print(f"| {arm} | " + " | ".join(nll) + f" | {kv} | {cell(acc(p, 'kvret_64k'), acc(g, 'kvret_64k'))} | "
              f"{cell(acc(p, 'passkey_64k'), acc(g, 'passkey_64k'))} | {vt} |")


if __name__ == "__main__":
    main()
