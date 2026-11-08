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
- Repeats in one batch: every benchmark loops ``for i in range(n_repeat): outputs =
  self.compute(...)`` and reads the outputs only after the loop, so each repeat waited for its
  own slowest 32k-token answer (AIME: 30 prompts x 10 repeats, ~6 min each, the engine mostly
  idle). ``BaseBenchmark.compute`` now defers: it queues the instances and returns a lazy result;
  the first read generates every queued call in one ``generate_until`` (lm-eval's vLLM backend
  keeps one SamplingParams per request, so each repeat keeps its own seed). Only for a
  single-process model (world_size 1, evalchemy's TP engine on MN5); EVALCHEMY_BATCH_REPEATS=0
  restores the per-call generation.

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


TASK_REL = "evalchemy/eval/task.py"
TASK_ORIG_MD5 = "ca1e7b89dcf7f5dc77c491cb319dfbd0"
TASK_ANCHOR = "SAFETY_BUFFER_TOKENS = 16\n"
TASK_DEFER = '''

# PATCH(oellm MN5, 2026-11-07): compute() calls are deferred and generated together at the first
# read of any of their outputs -- the n_repeat loops of the benchmarks then run as one batch.
class _DeferredOutputs:
    """The outputs of one compute() call; reading them generates every pending call at once."""

    def __init__(self, owner, model, start, n):
        self._owner, self._model, self._start, self._n = owner, model, start, n

    def _get(self):
        _flush_deferred(self._owner, self._model)
        return self._owner._oellm_done[self._start:self._start + self._n]

    def __len__(self):
        return self._n

    def __iter__(self):
        return iter(self._get())

    def __getitem__(self, i):
        return self._get()[i]


def _defer(owner, model, inputs):
    done = owner.__dict__.setdefault("_oellm_done", [])
    pending = owner.__dict__.setdefault("_oellm_pending", [])
    start = len(done) + len(pending)
    pending.extend(inputs)
    return _DeferredOutputs(owner, model, start, len(inputs))


def _flush_deferred(owner, model):
    pending = owner.__dict__.get("_oellm_pending")
    if pending:
        owner._oellm_pending = []
        logging.getLogger(owner.__class__.__name__).info(
            "generating %d deferred instance(s) in one batch", len(pending))
        owner._oellm_done.extend(model.generate_until(pending))
'''
TASK_BEFORE = """        for instance in inputs:
            instance.task_name = task_name

        if model.world_size > 1 and do_slice:"""
TASK_AFTER = """        for instance in inputs:
            instance.task_name = task_name

        if model.world_size == 1 and os.environ.get("EVALCHEMY_BATCH_REPEATS", "1") != "0":
            return _defer(self, model, inputs)  # PATCH(oellm MN5): one batch for all repeats

        if model.world_size > 1 and do_slice:"""


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


def patch_task(stage: Path, check: bool) -> str:
    target = stage / TASK_REL
    text = target.read_text()
    if TASK_AFTER in text:
        return "evalchemy task.py: already patched"
    if md5(target) != TASK_ORIG_MD5 or text.count(TASK_ANCHOR) != 1 or text.count(TASK_BEFORE) != 1:
        raise SystemExit(f"evalchemy task.py: {target} is not the reviewed original")
    if check:
        return "evalchemy task.py: would patch"
    shutil.copy2(target, str(target) + ".orig")
    text = text.replace(TASK_ANCHOR, TASK_ANCHOR + TASK_DEFER, 1).replace(TASK_BEFORE, TASK_AFTER, 1)
    target.write_text(text)
    return "evalchemy task.py: patched"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", type=Path, default=STAGE)
    p.add_argument("--check", action="store_true", help="report only, change nothing")
    a = p.parse_args()
    for fn in (patch_humaneval, patch_lcb, patch_task):
        print(fn(a.stage, a.check), flush=True)


if __name__ == "__main__":
    sys.exit(main())
