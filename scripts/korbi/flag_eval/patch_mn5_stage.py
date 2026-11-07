#!/usr/bin/env python3
"""Patch the MN5 oellm-eval stage's evalchemy checkout for the code benchmarks (idempotent).

- HumanEval: the grader of submodules/oellm-eval/containers/patches/humaneval_execution.py (the one
  the JUPITER launcher binds over the image's file): a sample whose code outlives the outer
  deadline (default timeout + 60 s) counts as 'timed out' instead of aborting the whole HumanEval
  task. The MN5 stage had the + 10 s version that raised: the 13k evalchemy group lost HumanEval
  at sample 46/164 (2026-11-06).
- LiveCodeBench: the question loading's ``shard.map(..., num_proc=os.cpu_count())`` pickles its
  lambda with dill for 128-160 processes; pylibs_oellm's dill 0.4.1 cannot pickle numpy 2.1's
  ``numpy._ArrayFunctionDispatcher`` (PicklingError before any generation). The map runs in
  process unless EVALCHEMY_LCB_NUM_PROC > 0.

Each file is changed only if it is the exact reviewed original (md5); the original is kept as
<file>.orig. Run on the MN5 login node from a synced snapshot:

    python3 scripts/korbi/flag_eval/patch_mn5_stage.py [--stage <stage dir>] [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STAGE = Path("/gpfs/projects/ehpc533/users/korbip/hybrid_v019/oellm_mn5/oellm_mn5_stage")

HE_REL = "evalchemy/eval/chat_benchmarks/HumanEval/human_eval/execution.py"
HE_ORIG_MD5 = "d4dcb1a0e2a1dca44c77ec68bf871366"
HE_PATCH = REPO / "submodules/oellm-eval/containers/patches/humaneval_execution.py"

LCB_REL = "evalchemy/eval/chat_benchmarks/LiveCodeBench/eval_instruct.py"
LCB_ORIG_MD5 = "7aae15b5c817c0a24eec7ccbfe2ef0d6"
LCB_BEFORE = "                num_proc=cpu_count,\n"
LCB_AFTER = (
    "                # PATCH(oellm MN5, 2026-11-07): in process by default -- dill cannot pickle\n"
    "                # this lambda's numpy globals (numpy 2.1 _ArrayFunctionDispatcher)\n"
    '                num_proc=int(os.environ.get("EVALCHEMY_LCB_NUM_PROC", "0")) or None,\n'
)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def patch_humaneval(stage: Path, check: bool) -> str:
    target = stage / HE_REL
    want = md5(HE_PATCH)
    have = md5(target)
    if have == want:
        return "HumanEval: already patched"
    if have != HE_ORIG_MD5:
        raise SystemExit(f"HumanEval: {target} is neither the reviewed original nor the patch ({have})")
    if check:
        return "HumanEval: would patch"
    shutil.copy2(target, str(target) + ".orig")
    shutil.copyfile(HE_PATCH, target)
    assert md5(target) == want
    return "HumanEval: patched"


def patch_lcb(stage: Path, check: bool) -> str:
    target = stage / LCB_REL
    text = target.read_text()
    if LCB_AFTER in text:
        return "LiveCodeBench: already patched"
    if md5(target) != LCB_ORIG_MD5 or text.count(LCB_BEFORE) != 1:
        raise SystemExit(f"LiveCodeBench: {target} is not the reviewed original")
    if check:
        return "LiveCodeBench: would patch"
    shutil.copy2(target, str(target) + ".orig")
    target.write_text(text.replace(LCB_BEFORE, LCB_AFTER, 1))
    return "LiveCodeBench: patched"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", type=Path, default=STAGE)
    p.add_argument("--check", action="store_true", help="report only, change nothing")
    a = p.parse_args()
    for fn in (patch_humaneval, patch_lcb):
        print(fn(a.stage, a.check), flush=True)


if __name__ == "__main__":
    sys.exit(main())
