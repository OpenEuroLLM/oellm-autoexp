#!/usr/bin/env python3
"""Verify a diagnostics_smoke_130M run actually produced CORRECT diagnostics.

A smoke test that only checks "the job did not crash" would pass with every
per-layer number silently wrong — a bad shard range or a missing TP-duplicate
filter changes values, not exit codes. So this checks the numbers.

THE DECISIVE CHECK is `diag/grad_norm/total_check` against Megatron's own
`grad-norm`. The per-layer norms are sums of squares over pieces that are
supposed to partition the gradient exactly once; recombining them must reproduce
the global norm Megatron computes by a completely independent path.

    ratio 1.00   shard ranges and duplicate filtering are both right
    ratio 1.41   a factor of 2 in norm^2 is being double-counted (sqrt(2))
    ratio 2.00   an axis (TP or DP) is fully double-counted
    ratio 0.71   half the gradient is being dropped

Reads tensorboard event files, because wandb is offline on JUPITER.

    python scripts/korbi/check_diagnostics_smoke.py <tensorboard_dir> [...]
"""

import argparse
import glob
import os
import sys
from collections import defaultdict


def load_scalars(tb_dir):
    """Tag -> {step: value} from every event file under tb_dir."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out = defaultdict(dict)
    files = glob.glob(os.path.join(tb_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not files:
        raise SystemExit(f"no tensorboard event files under {tb_dir}")
    for path in files:
        acc = EventAccumulator(path, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            for ev in acc.Scalars(tag):
                out[tag][ev.step] = ev.value
    return out


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def verify(tb_dir, num_layers, tol):
    print(f"\n=== {tb_dir} ===")
    s = load_scalars(tb_dir)
    diag = {t: v for t, v in s.items() if t.startswith("diag/")}
    if not diag:
        print("  [FAIL] no diag/* scalars at all — were the --diag-* flags set?")
        return False
    print(f"  {len(diag)} diag/* series, {len(s)} total")
    ok = True

    # --- the decisive one ---------------------------------------------------
    total = s.get("diag/grad_norm/total_check", {})
    ref = s.get("grad-norm", {})
    shared = sorted(set(total) & set(ref))
    if not shared:
        ok &= check("grad-norm recombination", False, "no iteration has both series")
    else:
        worst_step, worst_ratio = None, 1.0
        for step in shared:
            if ref[step] == 0:
                continue
            r = total[step] / ref[step]
            if abs(r - 1.0) > abs(worst_ratio - 1.0):
                worst_step, worst_ratio = step, r
        ok &= check(
            "per-layer grad norms recombine to the global grad-norm",
            abs(worst_ratio - 1.0) <= tol,
            f"worst ratio {worst_ratio:.6f} at iteration {worst_step} "
            f"over {len(shared)} iterations (tol {tol})",
        )

    # --- coverage: every layer reported, no gaps ----------------------------
    for family in ("input_norm", "pre_mlp_norm", "q_norm", "k_norm"):
        layers = {
            int(t.split("/layer_")[1].split("/")[0])
            for t in diag
            if t.startswith(f"diag/gain/{family}/layer_")
        }
        if not layers:
            print(f"  [skip] gain family {family} absent from this model")
            continue
        missing = sorted(set(range(num_layers)) - layers)
        ok &= check(
            f"gain/{family} covers all {num_layers} layers",
            not missing,
            f"missing {missing}" if missing else f"layers 0..{max(layers)}",
        )

    # `layer_NN` only — `diag/grad_norm/layer_max`, `layer_min` and `layer_ratio`
    # share the prefix and are summaries, not per-layer series.
    gl = {
        int(t.rsplit("_", 1)[1])
        for t in diag
        if t.startswith("diag/grad_norm/layer_") and t.rsplit("_", 1)[1].isdigit()
    }
    if gl:
        missing = sorted(set(range(num_layers)) - gl)
        ok &= check(
            f"grad_norm covers all {num_layers} layers",
            not missing,
            f"missing {missing}" if missing else f"layers 0..{max(gl)}",
        )

    # --- sanity: gains start at 1.0 and have not moved far in ~60 iterations --
    means = [
        v for t, vv in diag.items() if t.endswith("/mean") and "/gain/" in t for v in vv.values()
    ]
    if means:
        ok &= check(
            "norm gains are near their init value of 1.0",
            all(0.5 < m < 1.6 for m in means),
            f"range [{min(means):.4f}, {max(means):.4f}] over {len(means)} points",
        )

    # --- activations ---------------------------------------------------------
    rms = [v for t, vv in diag.items() if t.endswith("/rms_denom") for v in vv.values()]
    if rms:
        ok &= check(
            "RMSNorm denominators are finite and positive",
            all(0 < r < 1e6 for r in rms),
            f"range [{min(rms):.4f}, {max(rms):.4f}] over {len(rms)} points",
        )
        # COVERAGE, not just sanity. A hook only fires on the rank owning the
        # layer and the writers live on the last rank, so without a reduction
        # across pipeline stages this silently reports ONE STAGE ONLY and every
        # value in it still looks healthy. That is exactly what job 1583194
        # (PP=2) did: layers 9-17 present, 0-8 missing.
        for tag in ("input_norm", "pre_mlp_norm"):
            al = {
                int(t.split("/layer_")[1].split("/")[0])
                for t in diag
                if t.startswith(f"diag/act/{tag}/layer_")
            }
            if not al:
                continue
            missing = sorted(set(range(num_layers)) - al)
            ok &= check(
                f"act/{tag} covers all {num_layers} layers",
                not missing,
                f"missing {missing} — activations are not reduced across pipeline stages"
                if missing
                else f"layers 0..{max(al)}",
            )
    else:
        ok &= check("activation hooks fired", False, "no */rms_denom series — hooks never ran")

    # --- non-finite ----------------------------------------------------------
    for tag in ("diag/nonfinite/weight", "diag/nonfinite/grad", "diag/act/nonfinite"):
        vals = list(s.get(tag, {}).values())
        if vals:
            ok &= check(f"{tag} is zero", max(vals) == 0, f"max {max(vals)}")

    # --- clip ---------------------------------------------------------------
    if "diag/clip/max_streak" in s:
        streaks = list(s["diag/clip/max_streak"].values())
        gn = list(s.get("diag/grad_norm/mean", {}).values())
        check(
            "clip stats present (informational, not a pass/fail)",
            True,
            f"max streak {max(streaks):.0f}, mean grad-norm {sum(gn) / len(gn):.4f}"
            if gn
            else "no grad-norm mean",
        )
    else:
        ok &= check("clip event series present", False, "diag/clip/* missing")

    return ok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("tb_dirs", nargs="+", help="tensorboard directories, one per sweep arm")
    ap.add_argument("--num-layers", type=int, default=18)
    ap.add_argument(
        "--tol",
        type=float,
        default=2e-3,
        help="allowed relative deviation of total_check from grad-norm",
    )
    args = ap.parse_args()

    all_ok = True
    for d in args.tb_dirs:
        all_ok &= verify(d, args.num_layers, args.tol)
    print("\n" + ("ALL ARMS PASS" if all_ok else "FAILURES ABOVE"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
