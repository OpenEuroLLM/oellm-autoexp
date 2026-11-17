#!/usr/bin/env python3
"""Check a rendered *control* command before it is enqueued: the vanilla v4 continuation.

The control must be the v4a V arm's twin except for the intervention, so this asserts on the render:

  1. vanilla stack: the 98-layer pattern, no ``V`` layer, no ``--ckda-branch*`` / ``--kv-share*`` /
     ``--attention-head-temperature`` (the V insertion is the only difference to the arm)
  2. the main run's own 35900 rolling save as ``--load`` (never the surgery / V checkpoint)
  3. the 128 x mbs 2 == 256 x mbs 1 pairing: gbs 1024, seq 65536, reference sampler 512 / 1
  4. the session's save plan: exit 36900, extra saves 35999 / 36400 / 36900, under the v4_cont root
  5. every dataset cache-key flag equal to the *prebuild's* render, so the run finds the built index
     instead of building it (which twice OOMed a rank at 512 ranks, mn5-pool-p264.1261/.1264)

    check_control_render.py <control-render> <prebuild-render>

Exit 0 = enqueue it; 3 = do not.
"""

from __future__ import annotations

import os
import re
import sys

VANILLA = (
    "*ENLELELELELELELE*ELELELELELELELE*ELELELELELELELEN"
    "*ELELELELELELELE*ELELELELELELELE*ELELELELELELELE"
)
KEY_FLAGS = [
    "--data-args-path",
    "--data-cache-path",
    "--seq-length",
    "--split",
    "--data-sharding",
    "--dataloader-type",
    "--global-batch-size",
    "--eval-iters",
    "--eval-interval",
    "--train-iters",
    "--tokenizer-model",
    "--seed",
    "--num-dataset-builder-threads",
]
FORBIDDEN = ("--ckda-branch", "--kv-share-layers", "--kv-share-source-layers", "--attention-head-temperature")


def values(path: str) -> dict[str, str]:
    text = open(path, errors="replace").read().replace("\\\n", " ")
    out: dict[str, str] = {}
    for m in re.finditer(r"(--[A-Za-z0-9-]+)(?:[ =])(\"[^\"]*\"|'[^']*'|\S*)", text):
        out.setdefault(m.group(1), (m.group(2) or "").strip("\"'"))
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 3
    render, prebuild = argv[1], argv[2]
    v = values(render)
    problems: list[str] = []

    pattern = v.get("--hybrid-layer-pattern", "")
    if pattern != VANILLA:
        problems.append(f"--hybrid-layer-pattern is not the vanilla 98-layer stack: {pattern!r}")
    if "V" in pattern:
        problems.append("the pattern contains a V (kvshare) layer; that is the intervention")

    for flag in FORBIDDEN:
        present = [k for k in v if k.startswith(flag)]
        if present:
            problems.append(f"branch/temperature flag present in a control: {' '.join(present)}")

    # --expect-load-substring: the experiment segment loads the main run's 35900 rolling save
    # ("main_mhc4/...ckpt_rolling"); an open-ended continuation instead resumes from its own ckpt/.
    load = v.get("--load", "")
    want_load = os.environ.get("EXPECT_LOAD_SUBSTRING", "main_mhc4/ambitious_e360_eng80_v4_2ph_hd256/ckpt_rolling")
    if want_load not in load:
        problems.append(f"--load {load!r} does not contain {want_load!r}")

    expect = {
        "--global-batch-size": "1024",
        "--micro-batch-size": "1",
        "--seq-length": "65536",
        "--data-sampler-reference-dp-size": "512",
        "--data-sampler-reference-micro-batch-size": "1",
        # validation as the original v4 training ran it until 35900 (main run's log, .0987)
        "--eval-interval": "1000",
        "--eval-iters": "10",
        "--eval-global-batch-size": "1024",
    }
    # an open-ended continuation has no --exit-interval (ALLOW_OPEN_ENDED=1)
    if os.environ.get("ALLOW_OPEN_ENDED") != "1":
        expect["--exit-interval"] = "36900"
    elif "--exit-interval" in v:
        problems.append(f"--exit-interval is set ({v['--exit-interval']}) but the run is meant to be open-ended")

    for flag, want in expect.items():
        got = v.get(flag, "<absent>")
        if got != want:
            problems.append(f"{flag} is {got!r}, expected {want!r}")

    # full_validation is what makes the valid index size independent of eval_iters, so the validation
    # prebuild matches this run by construction; without it the valid key moves with eval_interval
    if "--full-validation" not in v:
        problems.append("--full-validation is absent; the original run had it and the valid index key depends on it")

    save = v.get("--save", "")
    if "v4_cont" not in save:
        problems.append(f"--save is outside the v4_cont root: {save!r}")

    raw = open(render, errors="replace").read().replace("\\\n", " ")
    if os.environ.get("ALLOW_OPEN_ENDED") == "1":
        # an open-ended run has no fixed save steps; it must instead carry a rolling safety net and a
        # periodic persistent save (otherwise a host end could lose everything since the start)
        for flag, want in (("--save-interval", "1000"), ("--non-persistent-save-interval", "100")):
            got = v.get(flag, "<absent>")
            if got != want:
                problems.append(f"{flag} is {got!r}, expected {want!r} for an open-ended run")
    else:
        for step in ("35999", "36400", "36900"):
            if step not in raw:
                problems.append(f"save step {step} missing from the render")

    pre = values(prebuild)
    print(f"{'cache-key flag':34s} {'control':44s} prebuild")
    for key in KEY_FLAGS:
        a, b = v.get(key, "<absent>"), pre.get(key, "<absent>")
        mark = "ok  " if a == b else "DIFF"
        print(f"{key:34s} {mark} {a[:40]:44s} {b[:40]}")
        if a != b:
            problems.append(f"{key} differs from the prebuild: control={a!r} prebuild={b!r}")

    if problems:
        print("\nREFUSED:")
        for p in problems:
            print("  -", p)
        return 3
    print("\nOK: vanilla control, main 35900 load, paired geometry, same cache key as the prebuild")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
