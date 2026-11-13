#!/usr/bin/env python3
"""Build the v8ml datamix for JUPITER on the COMPLETE flag collection.

Input : the official JUPITER v8ml mix
        /e/data1/datasets/playground/mmlaion/oellm_testing/datamix/datamix_v8ml_jsc.txt
        (80 % v8 English + 20 % flag-ml, tokenizer-256k-agent; weight/prefix per line).
        Its flag sources point at oellm_testing/flag-decon-v2-agenttok/..., of which only
        ~1/3 is on disk on JUPITER.
Output: datamix_v8ml_flagfull_jsc.txt next to this script, for Megatron --data-args-path
        (whitespace-separated "weight prefix" pairs; no comments allowed in that file).

Mapping:
  oellm_testing/flag-decon-v2-agenttok/<source>/...  ->  FLAG/<source>/...   (same layout)
  everything else (v8-decon-agenttok: hftok, vtok, mv1sub) -> kept as is
  entries whose .bin/.idx are missing are dropped; remaining weights rescaled to sum to 1.

Caveats (see smoke_test/README.md):
  - FLAG is the ORIGINAL flag collection, not the decontaminated flag-decon-v2-agenttok
    copy: same sources/weights, but benchmark-overlapping documents are not removed.
  - Do NOT validate on oellm_testing/v8ml-valid with this mix: those held-out documents
    are still contained in the original flag collection.

Reading FLAG needs the e-sta-openeurollm group. This script only reads data; it writes
just its own output file. Run from a fresh login (or via `sg e-sta-openeurollm -c ...`):
    python3 make_datamix_v8ml_flagfull.py
"""

import os
import sys

T = "/e/data1/datasets/playground/mmlaion/oellm_testing"
SRC_MIX = f"{T}/datamix/datamix_v8ml_jsc.txt"
FLAG = "/e/fscratch/e-sta-openeurollm/training_v2/training/collection/flag"
DECON_PREFIX = f"{T}/flag-decon-v2-agenttok/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datamix_v8ml_flagfull_jsc.txt")


def main() -> int:
    if not os.access(FLAG, os.R_OK | os.X_OK):
        print(f"Cannot read {FLAG} (need group e-sta-openeurollm)", file=sys.stderr)
        return 1

    kept, dropped = [], []
    for line in open(SRC_MIX):
        if not line.strip():
            continue
        weight, prefix = line.split()
        weight = float(weight)
        if prefix.startswith(DECON_PREFIX):
            prefix = os.path.join(FLAG, prefix[len(DECON_PREFIX):])
        if os.path.isfile(prefix + ".bin") and os.path.isfile(prefix + ".idx"):
            kept.append((weight, prefix))
        else:
            dropped.append((weight, prefix))

    total = sum(w for w, _ in kept)
    with open(OUT, "w") as f:
        for weight, prefix in kept:
            f.write(f"{weight / total:.10f} {prefix}\n")

    print(f"wrote {OUT}")
    print(f"kept    {len(kept)} entries, original weight {total:.4f}% -> rescaled to 1")
    print(f"dropped {len(dropped)} entries, weight {sum(w for w, _ in dropped):.4f}%")
    for weight, prefix in dropped:
        print(f"   - {weight:.4f}%  {prefix.replace(T + '/', '')}")
    n_flag = sum(1 for _, p in kept if p.startswith(FLAG))
    print(f"from full flag: {n_flag} entries; from oellm_testing (v8-only): {len(kept) - n_flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
