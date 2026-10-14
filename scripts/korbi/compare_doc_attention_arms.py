#!/usr/bin/env python3
"""Read off the packed_doc_attention_1node validation.

Two questions, in order of importance:

1. Do the two CONTROL arms produce bit-identical losses? `ctrl-nomask` runs full
   causal attention; `ctrl-densemask` sets --reset-attention-mask and lets the
   dataloader build the dense [b, 1, s, s] mask. If TE really does discard that
   mask (it only reads attention_mask for padding/arbitrary mask types), the two
   arms must agree to every digit Megatron prints. Any divergence falsifies the
   whole analysis in CROSS_DOC_ATTENTION.md, and nothing else here matters.

2. Does the `packed` arm differ, and what does it cost? It should part from the
   controls within a few steps (it is masking attention the controls do not), and
   step time should IMPROVE slightly -- varlen attention does strictly fewer
   FLOPs than dense causal.

NOTE ON TFLOP/s: Megatron's FLOP formula assumes full causal attention, so the
packed arm's reported TFLOP/s reads high relative to work actually done. This
script deliberately reports ms/iteration and tok/s/GPU, never TFLOP/s.

USAGE
-----
    # on jupiter, after the three arms finish
    python scripts/korbi/compare_doc_attention_arms.py \
        --root /e/project1/e-sta-openeurollm/pre_production_training

    # explicit job ids (when a run dir holds logs from several attempts)
    python scripts/korbi/compare_doc_attention_arms.py \
        --root ... --job ctrl-nomask=1492415 --job packed=1492416
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# The three arms, in the order they should be read.
ARMS = ("ctrl-nomask", "ctrl-densemask", "packed")
# Globbed, not formatted: the run directory carries the batch size and lr, which
# change when the config is retuned (gbs64 -> gbs2048 during bring-up).
RUN_DIR_GLOB = "oellm_32b_dense_pda-{arm}_*"

# ` iteration       7/      20 | ... | lm loss: 1.089331E+01 | ... `
ITER_RE = re.compile(r"iteration\s+(\d+)\s*/\s*\d+")
LOSS_RE = re.compile(r"lm loss:\s*([0-9.eE+-]+)")
MS_RE = re.compile(r"elapsed time per iteration \(ms\):\s*([0-9.]+)")
TOK_RE = re.compile(r"Tok/s/GPU\):\s*([0-9.]+)")
# Anything that means the arm did not actually produce a comparable run.
FAIL_RE = re.compile(
    r"Traceback|AssertionError|RuntimeError|CUDA out of memory|srun: error|NCCL error"
)


def find_log(root: Path, arm: str, job: str | None, run_glob: str = RUN_DIR_GLOB) -> Path | None:
    """Newest slurm log for an arm, or the one matching an explicit job id.

    Searches every run directory for the arm, since a retune changes the
    directory name and leaves the older, failed one in place.
    """
    log_dirs = [d / "logs" for d in root.glob(run_glob.format(arm=arm)) if d.is_dir()]
    if job:
        for log_dir in log_dirs:
            candidate = log_dir / f"slurm-{job}.log"
            if candidate.is_file():
                return candidate
        return None
    logs = [p for log_dir in log_dirs for p in log_dir.glob("slurm-*.log")]
    return max(logs, key=lambda p: p.stat().st_mtime) if logs else None


def parse(path: Path) -> tuple[dict[int, dict[str, float]], str | None]:
    """Return {iteration: {loss, ms, tok}} plus the first failure line, if
    any."""
    iterations: dict[int, dict[str, float]] = {}
    failure = None
    for line in path.read_text(errors="replace").splitlines():
        if failure is None and FAIL_RE.search(line):
            failure = line.strip()[:200]
        it, loss = ITER_RE.search(line), LOSS_RE.search(line)
        if not (it and loss):
            continue
        record = {"loss": float(loss.group(1))}
        if ms := MS_RE.search(line):
            record["ms"] = float(ms.group(1))
        if tok := TOK_RE.search(line):
            record["tok"] = float(tok.group(1))
        iterations[int(it.group(1))] = record
    return iterations, failure


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _control_check(data, noise_ratio):
    """PASS/FAIL on whether the dense dataloader mask is inert. Returns a
    verdict code.

    NOT an exact-equality test: the arms run on different nodes and flash attention's
    backward is non-deterministic unless NVTE_ALLOW_NONDETERMINISTIC_ALGO=0, so two runs of
    the SAME config drift apart once gradients accumulate. Observed: the controls were
    bit-identical at iterations 1-2 and differed by 1E-05 by iteration 3, while the packed
    arm differed by ~1E-03 from iteration 1. The discriminating question is one of
    MAGNITUDE -- if the dense mask reached the kernel it would show at iteration 1, at the
    packed arm's scale.
    """
    a, b = data.get("ctrl-nomask", {}), data.get("ctrl-densemask", {})
    p = data.get("packed", {})
    ctrl_shared = sorted(set(a) & set(b))
    packed_shared = sorted(set(a) & set(p))

    print("\n" + "=" * 78)
    if not ctrl_shared:
        print("CONTROL CHECK  INCONCLUSIVE - the two control arms share no iterations.")
        print("=" * 78)
        return 1

    ctrl_delta = max(abs(a[i]["loss"] - b[i]["loss"]) for i in ctrl_shared)
    first = ctrl_shared[0]
    first_delta = abs(a[first]["loss"] - b[first]["loss"])
    packed_delta = (
        max(abs(a[i]["loss"] - p[i]["loss"]) for i in packed_shared) if packed_shared else None
    )
    print(
        f"  control spread (nomask vs densemask) : {ctrl_delta:.3E}  over {len(ctrl_shared)} iters"
    )
    print(f"  control spread at iteration {first:<9}: {first_delta:.3E}")
    if packed_delta is not None:
        print(f"  packed effect  (nomask vs packed)    : {packed_delta:.3E}")

    if packed_delta is None:
        print("\nCONTROL CHECK  INCONCLUSIVE - no packed arm to compare the scale against.")
        verdict = 1
    elif ctrl_delta == 0.0 or packed_delta > noise_ratio * ctrl_delta:
        print(
            f"\nCONTROL CHECK  PASS - the controls agree to within run-to-run noise, while the\n"
            f"               packed arm moves the loss {packed_delta / ctrl_delta:.0f}x further"
            if ctrl_delta
            else "\nCONTROL CHECK  PASS - the controls agree exactly, the packed arm does not"
        )
        print(
            "               TE discards the dataloader mask: reset_attention_mask is inert,\n"
            "               and cu_seqlens genuinely reaches the kernel."
        )
        verdict = 0
    else:
        print(
            f"\nCONTROL CHECK  FAIL - the controls differ by {ctrl_delta:.3E}, not small relative\n"
            f"               to the packed effect ({packed_delta:.3E}). The dense mask may be\n"
            "               reaching the kernel; CROSS_DOC_ATTENTION.md would be wrong."
        )
        verdict = 2
    print("=" * 78)
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/e/project1/e-sta-openeurollm/pre_production_training"),
        help="directory holding the oellm_32b_dense_pda-* run directories",
    )
    parser.add_argument(
        "--job",
        action="append",
        default=[],
        metavar="ARM=JOBID",
        help="pin an arm to a specific slurm job id (repeatable)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="iterations to skip when averaging step time (default: 3)",
    )
    parser.add_argument(
        "--arms",
        default=",".join(ARMS),
        help=f"comma-separated arm names to read (default: {','.join(ARMS)})",
    )
    parser.add_argument(
        "--run-glob",
        default=RUN_DIR_GLOB,
        help="run-directory glob; {arm} is substituted (default: %(default)s)",
    )
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="ARM=LOSS",
        help="assert an arm's iteration-1 lm loss (repeatable). ONLY valid between runs "
        "with the SAME parallelism: initialize.py:415 offsets the seed by "
        "100 * pipeline_rank, so changing PP changes the initial weights and the losses "
        "are not comparable. Use --pair for a cross-PP check.",
    )
    parser.add_argument(
        "--expect-tol",
        type=float,
        default=1e-4,
        help="tolerance for --expect (default: %(default)s, chosen to sit well below the "
        "~5.8e-4 packed-vs-control signal so a wrong arm cannot pass)",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="CTRL:PACKED",
        help="compare a control arm against a packed arm sharing the same parallelism "
        "(repeatable). This is the cross-PP-safe check: the two arms have identical initial "
        "weights, so a NON-ZERO delta proves cu_seqlens reached every stage. Comparing "
        "absolute losses across different PP is meaningless (see --expect).",
    )
    args = parser.parse_args()

    pinned = dict(pair.split("=", 1) for pair in args.job)
    expected = {k: float(v) for k, v in (p.split("=", 1) for p in args.expect)}
    pairs = [tuple(p.split(":", 1)) for p in args.pair]

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    data: dict[str, dict[int, dict[str, float]]] = {}
    for arm in arms:
        log = find_log(args.root, arm, pinned.get(arm), args.run_glob)
        if log is None:
            print(f"  {arm:16s} NO LOG FOUND under {args.root}")
            continue
        iterations, failure = parse(log)
        data[arm] = iterations
        status = f"{len(iterations)} iterations"
        if failure:
            status += f"  FAILED: {failure}"
        print(f"  {arm:16s} {log.name:24s} {status}")

    if not data:
        print("\nNothing to compare.")
        return 1

    # ---- per-iteration losses ------------------------------------------------
    all_iters = sorted(set().union(*(set(d) for d in data.values())))
    print(f"\n{'iter':>5} " + " ".join(f"{arm:>16}" for arm in arms) + f" {'ctrl delta':>12}")
    for it in all_iters:
        cells = []
        for arm in arms:
            record = data.get(arm, {}).get(it)
            cells.append(f"{record['loss']:16.6E}" if record else f"{'-':>16}")
        a = data.get("ctrl-nomask", {}).get(it)
        b = data.get("ctrl-densemask", {}).get(it)
        delta = f"{abs(a['loss'] - b['loss']):12.3E}" if a and b else f"{'-':>12}"
        print(f"{it:5d} " + " ".join(cells) + f" {delta}")

    # ---- within-parallelism pair check (the PP>1 correctness test) -----------
    # The two arms of a pair share a parallelism, hence identical initial weights
    # (initialize.py:415 seeds per pipeline rank), so their delta is attributable to
    # cu_seqlens alone. At PP=1 that delta was 5.8E-04 at iteration 1.
    pair_verdict = 0
    if pairs:
        print(f"\n{'control':<18} {'packed':<18} {'iter':>5} {'delta':>11}  verdict")
        for ctrl_arm, packed_arm in pairs:
            ctrl_iters, packed_iters = data.get(ctrl_arm, {}), data.get(packed_arm, {})
            shared = sorted(set(ctrl_iters) & set(packed_iters))
            if not shared:
                print(f"{ctrl_arm:<18} {packed_arm:<18} {'-':>5} {'-':>11}  INCONCLUSIVE")
                pair_verdict = max(pair_verdict, 1)
                continue
            first = shared[0]
            delta = abs(ctrl_iters[first]["loss"] - packed_iters[first]["loss"])
            if delta == 0.0:
                verdict_text = "FAIL (packed is silently a control)"
                pair_verdict = 2
            elif delta > 1e-1:
                verdict_text = "FAIL (implausible - p2p fold?)"
                pair_verdict = 2
            else:
                verdict_text = "PASS (cu_seqlens is masking)"
            print(f"{ctrl_arm:<18} {packed_arm:<18} {first:5d} {delta:11.3E}  {verdict_text}")
        print(
            "\n  delta == 0        -> cu_seqlens never reached the middle stages.\n"
            "  delta ~1E-03      -> masking is active on every stage (PP=1 reference: 5.8E-04).\n"
            "  delta > 1E-01/NaN -> activations reinterpreted between stages; p2p fold wrong."
        )

    # ---- reference check (only valid within one parallelism) -----------------
    expect_verdict = 0
    if expected:
        print(f"\n{'arm':<20} {'iter-1 loss':>16} {'expected':>16} {'delta':>11}  verdict")
        for arm, want in expected.items():
            arm_iters = data.get(arm, {})
            record = arm_iters[min(arm_iters)] if arm_iters else None
            if not record:
                print(f"{arm:<20} {'no data':>16} {want:16.6E} {'-':>11}  INCONCLUSIVE")
                expect_verdict = max(expect_verdict, 1)
                continue
            delta = abs(record["loss"] - want)
            ok = delta <= args.expect_tol
            print(
                f"{arm:<20} {record['loss']:16.6E} {want:16.6E} {delta:11.3E}"
                f"  {'PASS' if ok else 'FAIL'}"
            )
            if not ok:
                expect_verdict = 2
        print(
            "\n  A packed arm landing on the CONTROL value means cu_seqlens never reached the\n"
            "  middle stages -- the run is silently a control. Anything else means the p2p\n"
            "  fold is wrong and activations are being reinterpreted between stages."
        )

    # ---- the load-bearing check ---------------------------------------------
    # NOT an exact-equality test. The arms run on different nodes and flash
    # attention's backward is non-deterministic unless NVTE_ALLOW_NONDETERMINISTIC_ALGO=0,
    # so two runs of the SAME config drift apart once gradients start accumulating.
    # Observed: the controls were bit-identical at iterations 1-2 and differed by
    # 1e-5 by iteration 3, while the packed arm differed by ~1e-3 from iteration 1.
    # So the discriminating question is one of MAGNITUDE: is the control-vs-control
    # spread far smaller than the packed-vs-control effect? If the dense mask were
    # reaching the kernel it would show up at iteration 1, at the packed arm's scale.
    NOISE_RATIO = 20.0

    # Only meaningful for the three default arms; other arm sets (PP tests, scatter A/B)
    # use --pair instead. Guarded rather than early-returned, so the cost report below
    # still runs for them.
    if {"ctrl-nomask", "ctrl-densemask"} <= set(arms):
        verdict = _control_check(data, NOISE_RATIO)
    else:
        verdict = 0

    # ---- cost ----------------------------------------------------------------
    print(f"\nstep time / throughput (iterations > {args.warmup})")
    baseline = None
    baseline_arm = None
    for arm in arms:
        records = [r for i, r in sorted(data.get(arm, {}).items()) if i > args.warmup]
        ms = mean([r["ms"] for r in records if "ms" in r])
        tok = mean([r["tok"] for r in records if "tok" in r])
        if ms is None:
            total = len(data.get(arm, {}))
            reason = (
                f"only {total} iterations, all <= --warmup {args.warmup}"
                if total and total <= args.warmup
                else "no timing lines in the log (needs log_throughput)"
            )
            print(f"  {arm:16s} {reason}")
            continue
        if baseline is None:
            baseline = ms
            baseline_arm = arm
        rel = (
            f"  {(ms / baseline - 1) * 100:+6.1f}% vs {baseline_arm}"
            if baseline and arm != baseline_arm
            else ""
        )
        tok_str = f"{tok:9.1f} tok/s/GPU" if tok else ""
        print(f"  {arm:16s} {ms:8.1f} ms/iter  {tok_str}{rel}")

    return max(verdict, expect_verdict)


if __name__ == "__main__":
    sys.exit(main())
