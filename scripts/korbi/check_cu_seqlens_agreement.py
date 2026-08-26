#!/usr/bin/env python3
"""Do all pipeline stages agree on the document boundaries?

Under --packed-doc-attention every pipeline stage reads the dataloader and derives
cu_seqlens itself. Nothing asserts at runtime that they arrive at the same answer, and
they cannot be made to check each other: a collective inside forward_step deadlocks
against the p2p activation chain (measured, job 1494386 -- see
_shared_packed_seq_params). If the stages ever disagreed, attention would be masked
inconsistently down the pipeline and nothing would complain.

cu_seqlens is a few dozen bytes, so the cheap way out is to just print it and compare
offline. Run training with

    backend.megatron.packed_doc_attention_log_cu_seqlens: 8

and every rank emits, for its first N calls to get_batch:

    [PDA] call=0 rank=3 pp=1 tp=1 vp=None max_seqlen=1994 cu_seqlens=[0, 1994, ...]

Ranks are comparable by CALL INDEX, not by wall-clock: the interleaved schedule's lookup
tables (get_schedule_table, schedules.py:1045) depend only on num_microbatches, the chunk
count and microbatch_group_size_per_vp_stage -- never on pipeline rank -- so the k-th call
is the same (chunk, microbatch) everywhere. Only the timing differs.

USAGE
    python scripts/korbi/check_cu_seqlens_agreement.py <slurm log> [...]

Exit status is the number of (call, vp) groups where the ranks disagreed.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# `dp=` is optional: it was added to the [PDA] line after this script existed, so logs from
# before that are still readable (see parse() for how dp is reconstructed for those).
LINE_RE = re.compile(
    r"\[PDA\] call=(?P<call>\d+) rank=(?P<rank>\d+) pp=(?P<pp>\d+) tp=(?P<tp>\d+) "
    r"(?:dp=(?P<dp>\d+) )?vp=(?P<vp>\S+) max_seqlen=(?P<max_seqlen>\d+) "
    r"cu_seqlens=(?P<cu>\[[^\]]*\])"
)


def parse(paths, tp_size=None, dp_size=None):
    """({(call, vp, dp): {(pp, tp, rank): (max_seqlen, cu_seqlens)}},
    saw_dp_field)

    GROUPED BY DATA-PARALLEL REPLICA, which the first version of this
    script did not do -- it compared every rank against every other and
    therefore reported a FAIL on any DP>1 run, i.e. on every production
    shape. Different DP replicas are SUPPOSED to see different
    documents; only ranks within one replica must agree.

    Measured, job 1498402 (TP=2 PP=2 VPP=2 on 8 GPUs, so DP=2): ranks
    {0,1,4,5} and {2,3,6,7} printed different cu_seqlens for all 8
    logged calls and the script called it a failure. Grouped by replica,
    all 16 (call, vp, dp) groups are internally identical -- the code
    was right and the check was wrong.

    dp comes from the log line when present. For older logs it can be
    reconstructed from the rank, because the default rank order is tp-
    cp-ep-dp-pp, so with cp=ep=1     dp = (rank // tp_size) % dp_size
    which is why --tensor-parallel-size / --data-parallel-size exist.
    Without either, every rank lands in one group -- correct only at
    DP=1, and warned about in main().
    """
    records = defaultdict(dict)
    saw_dp_field = False
    for path in paths:
        for line in path.read_text(errors="replace").splitlines():
            m = LINE_RE.search(line)
            if not m:
                continue
            rank = int(m["rank"])
            if m.groupdict().get("dp") is not None:
                dp = int(m["dp"])
                saw_dp_field = True
            elif tp_size and dp_size:
                dp = (rank // tp_size) % dp_size
            else:
                dp = 0
            key = (int(m["call"]), m["vp"], dp)
            who = (int(m["pp"]), int(m["tp"]), rank)
            records[key][who] = (int(m["max_seqlen"]), m["cu"])
    return records, saw_dp_field


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Only needed for logs written before the [PDA] line carried dp=. Together "
        "with --data-parallel-size it reconstructs the replica from the rank.",
    )
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=None,
        help="See --tensor-parallel-size. Omit both only when DP=1.",
    )
    args = parser.parse_args()

    missing = [p for p in args.logs if not p.is_file()]
    for p in missing:
        print(f"NOT A FILE  {p}")
    records, saw_dp_field = parse(
        [p for p in args.logs if p.is_file()],
        tp_size=args.tensor_parallel_size,
        dp_size=args.data_parallel_size,
    )

    if not records:
        print(
            "No [PDA] lines found. Was the run started with "
            "packed_doc_attention_log_cu_seqlens > 0?"
        )
        return 1

    replicas = sorted({dp for (_, _, dp) in records})
    stages = sorted({pp for group in records.values() for (pp, _, _) in group})
    print(f"pipeline stages seen: {stages}   data-parallel replicas: {replicas}")
    if len(stages) < 2:
        print(
            "WARNING: only one pipeline stage in these logs -- this check is vacuous.\n"
            "         Point it at a PP>1 run, and make sure the log captures all ranks."
        )
    if not saw_dp_field and not args.data_parallel_size:
        # Every rank fell into one bucket. Fine at DP=1, and a guaranteed false FAIL
        # otherwise, because DP replicas legitimately read different documents.
        print(
            "NOTE: no dp= in these logs and --data-parallel-size not given, so all ranks\n"
            "      are compared as ONE replica. That is correct only at DP=1. If this run\n"
            "      had DP>1, re-run with --tensor-parallel-size/--data-parallel-size or the\n"
            "      disagreements below are just different replicas reading different data."
        )

    disagreements = 0
    for call, vp, dp in sorted(records):
        group = records[(call, vp, dp)]
        distinct = set(group.values())
        if len(distinct) == 1:
            ((max_seqlen, cu),) = distinct
            print(
                f"call={call:<3} vp={vp:<5} dp={dp:<2} {len(group):3d} ranks AGREE  "
                f"max_seqlen={max_seqlen} cu_seqlens={cu[:60]}"
            )
        else:
            disagreements += 1
            print(f"call={call:<3} vp={vp:<5} dp={dp:<2} DISAGREE across {len(group)} ranks:")
            for who in sorted(group):
                max_seqlen, cu = group[who]
                print(
                    f"    pp={who[0]} tp={who[1]} rank={who[2]}  "
                    f"max_seqlen={max_seqlen} cu_seqlens={cu[:80]}"
                )

    print()
    if disagreements:
        print(
            f"FAIL: {disagreements} call(s) where the stages derived DIFFERENT document\n"
            "      boundaries. Per-stage local derivation is unsafe for this configuration;\n"
            "      attention is being masked inconsistently down the pipeline."
        )
    else:
        print(
            f"PASS: every rank agreed on cu_seqlens for all {len(records)} logged call(s).\n"
            "      Note this checks the calls that were logged, not the whole run."
        )
    return disagreements


if __name__ == "__main__":
    sys.exit(main())
