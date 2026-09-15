#!/usr/bin/env python3
"""List sweep index -> stage for an experiment config, without the slow
per-point Hydra resolve that `run_autoexp.py --dry-run` does.

`expand_sweep()` (used internally by the orchestrator) only expands the raw
sweep groups into parameter dicts in memory -- it doesn't run Hydra's
`compose()` per point like `resolve_sweep_with_dag` does. So this is a single
config load instead of one per sweep point, and finishes in ~seconds even for
sweeps with hundreds of points.

Usage:
    python tools/list_sweep_stages.py --config-name <name> [--stage-prefix decay]

Prints "<index> lr=<lr> gbsz=<gbsz> tokens=<N>BT <stage>" for every matching
point, and (when --stage-prefix is given) a ready-to-use `--array-subset ...`
range string on stderr.
"""

import argparse
import os
import sys

from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.config.schema import ConfigSetup
from oellm_autoexp.hydra_staged_sweep import expand_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("-C", "--config-dir", default="config")
    parser.add_argument(
        "--stage-prefix",
        default=None,
        help="Only list/collapse indices whose stage starts with this (e.g. 'decay').",
    )
    args = parser.parse_args()

    config_setup = ConfigSetup(
        pwd=os.path.abspath(os.curdir),
        config_name=args.config_name,
        config_dir=str(args.config_dir),
        overrides=[],
        monitor_state_dir="./monitor_state",
    )
    root = load_config_reference(config_setup=config_setup)
    points = expand_sweep(root.sweep)

    matched = []
    for point in points:
        stage = point.parameters.get("stage")
        if args.stage_prefix and not (
            isinstance(stage, str) and stage.startswith(args.stage_prefix)
        ):
            continue
        matched.append(point.index)
        params = point.parameters
        lr = params.get("backend.megatron.lr")
        gbsz = params.get("backend.megatron.global_batch_size")
        tokens = params.get("backend.megatron.aux.tokens")
        tokens_str = f"{tokens / 1e9:g}BT" if isinstance(tokens, (int, float)) else tokens
        print(f"{point.index:4d}  lr={lr!s:<7} gbsz={gbsz!s:<4} tokens={tokens_str!s:<6} {stage}")

    if args.stage_prefix and matched:
        matched.sort()
        ranges = []
        start = prev = matched[0]
        for idx in matched[1:]:
            if idx == prev + 1:
                prev = idx
                continue
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = idx
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        print("\n--array-subset " + ",".join(ranges), file=sys.stderr)


if __name__ == "__main__":
    main()
