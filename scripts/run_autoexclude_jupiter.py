#!/usr/bin/env python3
"""Curate the JUPITER node-exclusion list from SLURM's own record.

WHY THIS EXISTS (the half a config cannot do)
---------------------------------------------
config/job/node_catch.yaml harvests every node that a LOG LINE names. That is
genuinely useful, and for the probe campaign it is most of the story. But the
most expensive JUPITER failure mode produces no log line at all:

    Prolog[0]   = /usr/sbin/nhc
    PrologFlags = Alloc,Contain,DeferBatch,ForceRequeueOnFail

so a node that fails the health check requeues the WHOLE job before the batch
script ever runs. Job 1371705 did this five times in fifteen minutes and job
1374701 six times; every attempt is ExitCode 0:54, State REQUEUED, log length
zero. No monitor, no regex and no log-tailing loop can see it. The only record
is the per-attempt NodeList that `sacct --duplicates -X` keeps:

    1374701  2026-08-14T16:50:22  REQUEUED  0:54  1024  jpbo-006-[39,43,47],...
    1374701  2026-08-14T16:52:49  REQUEUED  0:54  1024  jpbo-006-[39,42-43,47],...
    ...

Correlating those lists is how the 2026-08-14 block of
config/exclude/jupiter_exclude_nodes.txt was built — by hand, over 35 attempts,
against a ~6.5-appearance null expectation. This script does that mechanically,
and cross-checks every candidate against the drain reason SLURM is already
reporting in `sinfo -R`.

Subcommands
-----------
  analyze   Report which nodes are over-represented in failing allocations,
            with corroborating NHC/drain reasons. Read-only.
  update    Same analysis, then APPEND the survivors to the exclusion file with
            their evidence as comments. Dry-run unless --apply.
  prune     Re-check the nodes already in the file against current node state
            and comment out the ones that are healthy again. Dry-run unless
            --apply. (The file is technical debt by construction — every stale
            entry permanently shrinks the pool a 512-node job can schedule on.)
  campaign  Run a node-catching experiment end to end: launch run_autoexp.py,
            wait for it, then analyze + update over exactly the jobs it created.

Typical use
-----------
    # burn in the fleet, then curate
    python scripts/run_autoexclude_jupiter.py campaign \\
        --config-name experiments/oellm_32b_dense/node_catch_n512 --apply

    # or curate after the fact, e.g. after a night of failed training runs
    python scripts/run_autoexclude_jupiter.py analyze --since 2026-08-14
    python scripts/run_autoexclude_jupiter.py update  --since 2026-08-14 --apply

    # before the production run: give the fleet its nodes back
    python scripts/run_autoexclude_jupiter.py prune

THREE KINDS OF EVIDENCE, KEPT SEPARATE ON PURPOSE
-------------------------------------------------
  DIRECT          a failure line, or a probe verdict, NAMED this node. One
                  occurrence is sufficient; nothing statistical about it.
  DIED UNDER US   SLURM drained or downed the node WHILE it was running one of
                  our failing jobs. Also sufficient on its own, and it needs no
                  log at all — which is what makes a job that died MIDWAY
                  attributable even when the monitor was not running, the log
                  was lost, or the job hung and died without printing anything.
                  See attribute_mid_run_deaths() for the timestamps this rests
                  on and the aftermath false positives it has to reject.
  STATISTICAL     the node merely appeared in failing allocations more often
                  than chance explains.

HOW THE STATISTICAL TEST WORKS, AND TWO RULES THAT LOOK RIGHT BUT ARE NOT
-------------------------------------------------------------------------
Rule 1, "flag anything appearing more than Nx its expected number of times",
fails on the fat tail of the binomial: with F failing draws of n nodes from a
fleet of N, appearances are Binomial(F, n/N), and in a 10-draw campaign at
128/960 about 4.5% of the fleet lands at 3x expectation by luck alone. Measured
on synthetic data with exactly ONE planted bad node, that rule returned 185
candidates out of 718 nodes seen.

Rule 2, the same thing with a proper significance bar on the APPEARANCE COUNT,
survives synthetic data and then fails on the real thing. Replaying the 35
attempts of the 2026-08-14 requeue loop, it flagged 232 nodes at p down to
1e-07 — entire racks (jpbo-036-*, jpbo-030-*) appearing in 22 of 25 draws
against an expectation of 4.2. Those racks are not broken; SLURM simply keeps
handing out the same low-weight nodes, so draws are nowhere near independent
samples of the fleet.

What this script actually tests is the CONDITIONAL FAILURE RATE: of the draws a
node was in, did more than its share fail, given the campaign's overall failure
rate? A node SLURM always picks appears in the successful draws too, so the
re-use bias cancels; a node that really breaks jobs is present in the failures
and absent from the successes. The tail probability is Bonferroni-corrected for
having looked at every node in the fleet and must come in under --alpha.

THE LIMIT OF THAT TEST, STATED PLAINLY: when EVERY draw failed it has no power
at all, because there is nothing successful to contrast against. That is the
2026-08-14 case (31 of 35 attempts failed, the rest cancelled), and it is why
that block of the exclusion file was ultimately built by intersecting the
counts with NHC DRAIN RECORDS rather than by arithmetic.

SO THE WRITE BAR IS ASYMMETRIC, and the asymmetry is the design:

  corroborated    >= --min-fails failing draws AND an independent health drain
                  reason from sinfo (NHC, ECC, link counters, unresponsive) ->
                  written, no significance bar. NHC is a SECOND INSTRUMENT, not
                  another slice of the same counts, and the costs are lopsided:
                  a wrong exclusion costs 1/5565 of the fleet, a missed one
                  costs a 512-node job. Demanding family-wise significance on
                  top of a hardware fault report is over-correction — measured
                  on 2026-08-15..20, the eight most suspicious nodes ALL had
                  hardware NHC reasons and not one of them cleared it.
  uncorroborated  nothing but our own counts -> must clear the fleet-corrected
                  conditional test. Hard on purpose; --include-uncorroborated
                  relaxes it to the raw p if you know what you are doing.

Administrative drains (planned shutdown, image deployment) do NOT corroborate —
see ADMIN_REASON_RE. Treating any drain as evidence added 20 healthy nodes of
racks 011-020 on the first real run.

Neither does a reason shared by a WHOLE RACK (--corroborate-max-nodes). NHC is a
second instrument only while it discriminates BETWEEN nodes; when 80 nodes in
four contiguous blocks all read "Node not responding for 3600 seconds", that is
one cell-level outage reported 80 times, and corroborating on it writes 80
entries for an event nobody expects to persist. Measured 2026-08-23 on the 32B
window: 80 of 84 candidates were exactly that. This is the third instance of the
same cap — DIRECT_PATTERNS caps how many nodes one log may attribute, and
attribute_mid_run_deaths() caps how many one job's drains may — and it is the
same principle each time: mass attribution is not attribution.

COROLLARY FOR RUNNING A CAMPAIGN: successful draws are not wasted draws, they
are what gives the failing ones meaning. And run the analysis while the
evidence is FRESH — sinfo reports today's drain state, so a node drained during
the campaign and resumed the next morning has lost its corroboration by then.
`campaign` does the analysis immediately for exactly this reason.

Run this on a JUPITER login node (needs sacct/sinfo/scontrol).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# MUST RESOLVE EXACTLY AS config/slurm/jupiter_autoexclude.yaml DOES, or this
# script curates one file while every job reads another and the two silently
# diverge. The six config sites all interpolate
#     ${oc.env:JUPITER_EXCLUDE_NODES,<SHARED_EXCLUDE_FILE>}
# so the env var wins here too. This used to be hardcoded to the repo copy,
# which was already a latent bug and became a real one the moment the fallback
# moved off the repo (2026-08-23).
SHARED_EXCLUDE_FILE = Path(
    "/e/project1/e-sta-openeurollm/production_training/jupiter_exclude_nodes.txt"
)
DEFAULT_EXCLUDE_FILE = Path(os.environ.get("JUPITER_EXCLUDE_NODES") or SHARED_EXCLUDE_FILE)

# sacct States that count as a failing ALLOCATION. REQUEUED is the important
# one: it is the prolog/NHC signature that has no log at all.
FAIL_STATES = {
    "REQUEUED",
    "NODE_FAIL",
    "FAILED",
    "BOOT_FAIL",
    "OUT_OF_MEMORY",
    "DEADLINE",
    "PREEMPTED",
}
# TIMEOUT is NOT a failure here: a 12 h wall on a chained training run is the
# expected way for a job to end. COMPLETED is the healthy case. CANCELLED is
# ambiguous (an operator scancel and a monitor CancelAction look identical), so
# it is neutral unless --cancelled-as-fail says otherwise.
OK_STATES = {"COMPLETED", "TIMEOUT"}

# Drain reasons that are ADMINISTRATIVE, not evidence of a fault. A node can be
# down for a planned shutdown or an image rollout and be in perfect health, so
# these must not count as corroboration — replaying the 2026-08-14 jobs, "any
# drain reason counts" promoted 20 nodes of racks 011-020 purely because they
# are down for "IT-1-2 Planned Shutdown" TODAY, two weeks after the failures.
ADMIN_REASON_RE = re.compile(
    r"planned shutdown|deployment of system image|maintenance|reboot|"
    r"software update|reserved|reservation|absent|expose to the public",
    re.IGNORECASE,
)

# Log-line shapes that ATTRIBUTE a fault to a specific node. Kept in sync with
# config/job/node_catch.yaml — see that file for why the slurmstepd
# "JOB x ON <node> CANCELLED" line is deliberately absent (it names the batch
# node, not the culprit).
# The third element caps how many DISTINCT nodes one log may attribute through
# that pattern. See the note on the task-exit pattern for why that cap exists.
DIRECT_PATTERNS: list[tuple[str, re.Pattern[str], int]] = [
    # A probe log legitimately names one node per failing node, so no cap.
    ("probe FAIL", re.compile(r"\[nodecheck\] FAIL ([\w-]+): (.*)"), 0),
    ("SLURM node failure", re.compile(r"Node failure on ([\w-]+)()"), 0),
    # THE ONE THAT NEEDS A CAP. A single dead node shows up as exactly one
    #     srun: error: jpbo-120-36: task 1011: Exited with exit code 1
    # and that is how jpbo-120-36 (wedged GSP firmware) was identified. But when
    # a DISTRIBUTED job dies, every rank exits nonzero and srun prints the line
    # for the whole allocation: measured over the 2026-08-18.. training logs,
    # this pattern named 830 DISTINCT NODES — i.e. the allocations, not the
    # culprits. So the hits are kept only when a log attributes at most this
    # many distinct nodes; beyond that the line is reporting a mass crash and
    # carries no attribution at all.
    (
        "task exited nonzero",
        re.compile(
            r"srun: error: ([a-z][\w-]+): tasks? [0-9]+[^:]*: (Exited with exit code [0-9]+)"
        ),
        3,
    ),
]


# ---------------------------------------------------------------------------
# shelling out
# ---------------------------------------------------------------------------
def run(cmd: list[str], *, timeout: float = 120.0) -> tuple[int, str]:
    """Run a command, returning (returncode, stdout).

    Never raises.
    """
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired:
        return 124, ""
    if proc.returncode != 0 and proc.stderr.strip():
        print(f"warning: {' '.join(cmd[:3])}... -> {proc.stderr.strip()[:200]}", file=sys.stderr)
    return proc.returncode, proc.stdout


def expand_hostlist(nodelist: str) -> list[str]:
    """Expand a SLURM nodelist ("jpbo-018-[33-34,36],jpbo-020-01") to names.

    Uses `scontrol show hostnames`, which is authoritative, and falls
    back to a local parser so the script still works off-cluster
    (replaying a --from-json dump, or under test).
    """
    nodelist = nodelist.strip()
    if not nodelist or nodelist in {"None assigned", "(null)"}:
        return []
    rc, out = run(["scontrol", "show", "hostnames", nodelist])
    if rc == 0 and out.strip():
        return [line.strip() for line in out.splitlines() if line.strip()]
    return _expand_hostlist_py(nodelist)


def _expand_hostlist_py(nodelist: str) -> list[str]:
    """Pure-python hostlist expansion (fallback for `scontrol show
    hostnames`)."""
    names: list[str] = []
    # Split on commas that are NOT inside brackets.
    parts: list[str] = []
    depth = 0
    current = ""
    for char in nodelist:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
            continue
        current += char
    if current:
        parts.append(current)

    for part in parts:
        part = part.strip()
        if not part:
            continue
        match = re.match(r"^(.*?)\[([^\]]*)\](.*)$", part)
        if not match:
            names.append(part)
            continue
        prefix, ranges, suffix = match.groups()
        for token in ranges.split(","):
            token = token.strip()
            if "-" in token:
                lo, hi = token.split("-", 1)
                width = len(lo)
                for value in range(int(lo), int(hi) + 1):
                    names.append(f"{prefix}{value:0{width}d}{suffix}")
            elif token:
                names.append(f"{prefix}{token}{suffix}")
    return names


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------
@dataclass
class Attempt:
    """One sacct allocation record — i.e. one draw of nodes."""

    job_id: str
    job_name: str
    start: str
    end: str
    state: str
    exit_code: str
    nnodes: int
    nodes: list[str]
    verdict: str  # "fail" | "ok" | "neutral"


@dataclass
class NodeEvidence:
    appearances: int = 0
    # Appearances in draws that ended either "fail" or "ok" — i.e. draws that
    # carry information. Neutral (CANCELLED) draws are counted in `appearances`
    # for the report but must not enter the test.
    informative: int = 0
    failures: int = 0
    expected_failures: float = 0.0
    direct: list[str] = field(default_factory=list)  # named by a log line
    # A health drain/down event that began while THIS node was running one of
    # our failing jobs. Independent of the log entirely — see
    # attribute_mid_run_deaths().
    timed: list[str] = field(default_factory=list)
    fail_jobs: list[str] = field(default_factory=list)
    slurm_state: str = ""
    slurm_reason: str = ""
    # THE test: P(at least `failures` of this node's own draws failed | the
    # campaign-wide failure rate). See analyze() for why this and not the
    # appearance count. `p_raw` is that probability for THIS node alone and is
    # what the ranking uses; `p_corrected` is Bonferroni-corrected for having
    # scanned the whole fleet and is what the write gate uses. They differ by
    # three orders of magnitude, so a small campaign ranks nodes sensibly while
    # still refusing to auto-exclude any of them.
    p_raw: float = 1.0
    p_corrected: float = 1.0
    # Diagnostic only: P(at least `failures` appearances among the failing draws
    # if the node were drawn at random from the fleet). Confounded by SLURM's
    # node ordering — reported, never acted on.
    p_appearance: float = 1.0
    # How many of the nodes we looked at carry THIS node's drain reason, and the
    # cap above which that reason stops corroborating. See the note in
    # corroborated() — a reason shared by a whole rack is one event, not N.
    reason_shared_by: int = 0
    reason_max_nodes: int = 0

    @property
    def ratio(self) -> float:
        if self.expected_failures <= 0:
            return float(self.failures)
        return self.failures / self.expected_failures

    @property
    def corroborated(self) -> bool:
        """SLURM independently says something is wrong with this node."""
        # FOUR traps here, all hit on real data:
        #  1. sinfo prints the literal string "none" for an empty reason, so a
        #     plain truthiness test marks EVERY node corroborated (measured:
        #     232/232 candidates on the 2026-08-14 replay).
        #  2. a drain STATE with an administrative reason is not evidence — see
        #     ADMIN_REASON_RE (that added 20 more nodes, all "Planned Shutdown").
        #  3. states carry flags: "drained$", "down*". Substring match, not ==.
        #  4. A REASON SHARED BY A WHOLE RACK IS ONE EVENT, NOT N FAULTS, and
        #     without a cap it corroborates N exclusions. Measured 2026-08-23 on
        #     the 32B window: 80 of 84 statistical candidates were four
        #     CONTIGUOUS blocks — jpbo-071-[1-32], jpbo-073-[1-16],
        #     jpbo-075-[17-32], jpbo-077-[1-16] — all reading the identical
        #     "Node not responding for 3600 seconds", i.e. cells that dropped off
        #     the network together. Writing them would have taken the file from
        #     26 to ~110 entries for an outage nobody expects to persist.
        #     Same shape as the srun task-exit cap in DIRECT_PATTERNS and the
        #     aftermath cap in attribute_mid_run_deaths(), for the same reason:
        #     mass attribution is not attribution.
        reason = self.slurm_reason.strip()
        if not reason or reason.lower() in {"none", "(null)"}:
            return False
        if ADMIN_REASON_RE.search(reason):
            return False
        if self.reason_max_nodes and self.reason_shared_by > self.reason_max_nodes:
            return False
        return any(flag in self.slurm_state.lower() for flag in ("drain", "down", "fail", "maint"))


def reason_key(reason: str) -> str:
    """Normalise a drain reason so the SAME fault on many nodes groups
    together.

    Digits are dropped because the node-specific parts of an NHC reason are the
    numeric ones — a PCI bus id, a free-byte count, a timeout in seconds — while
    the part naming the CHECK is not. Without this,
        "Node not responding for 3600 seconds"
        "Node not responding for 7200 seconds"
    read as two unrelated faults and each rack splits into groups small enough to
    slip under the cap. Truncated for the same reason attribute_mid_run_deaths()
    truncates: a long NHC reason carries a per-node tail.
    """
    return re.sub(r"\s+", " ", re.sub(r"\d+", "", reason)).strip().lower()[:60]


def binomial_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p).

    Exact; n is at most a few dozen here.
    """
    if k <= 0:
        return 1.0
    if n <= 0 or p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return sum(math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i)) for i in range(k, n + 1))


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------
def collect_attempts(args: argparse.Namespace) -> list[Attempt]:
    """Pull one record per allocation ATTEMPT out of sacct.

    `--duplicates` is what makes requeued attempts visible: without it
    sacct collapses a job to its final state and the five allocations
    that were requeued before it — the ones that actually identify the
    bad nodes — vanish. `-X` restricts output to allocations (no
    .batch/.extern steps).
    """
    fields = ["JobID", "JobName", "Start", "End", "State", "ExitCode", "NNodes", "NodeList"]
    cmd = ["sacct", "--duplicates", "-X", "-P", "--noheader", "-o", ",".join(fields)]
    if args.jobs:
        cmd += ["-j", args.jobs]
    else:
        cmd += ["-u", args.user, "-S", args.since]
        if args.until:
            cmd += ["-E", args.until]

    rc, out = run(cmd, timeout=300)
    if rc != 0:
        print(f"error: sacct failed (rc={rc}) — are you on a login node?", file=sys.stderr)
        return []

    attempts: list[Attempt] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        cols = line.split("|")
        if len(cols) < len(fields):
            continue
        job_id, job_name, start, end, state, exit_code, nnodes, nodelist = cols[: len(fields)]
        if args.name and args.name not in job_name:
            continue
        try:
            n = int(nnodes)
        except ValueError:
            n = 0
        if n < args.min_nodes:
            continue
        # "CANCELLED by 12345" -> "CANCELLED"
        base_state = state.split()[0] if state else ""
        if base_state in FAIL_STATES:
            verdict = "fail"
        elif base_state in OK_STATES:
            verdict = "ok"
        elif base_state == "CANCELLED" and args.cancelled_as_fail:
            verdict = "fail"
        else:
            verdict = "neutral"
        attempts.append(
            Attempt(
                job_id=job_id,
                job_name=job_name,
                start=start,
                end=end,
                state=state,
                exit_code=exit_code,
                nnodes=n,
                nodes=expand_hostlist(nodelist),
                verdict=verdict,
            )
        )
    return attempts


def collect_direct_hits(log_globs: list[str]) -> dict[str, list[str]]:
    """Scan log files for lines that NAME a faulty node.

    Redundant with the monitor when it was running (it appends these
    live), and essential when it was not — a campaign driven with
    --submit-and-exit, or a run whose monitor was killed, leaves the
    evidence only in the log.
    """
    hits: dict[str, list[str]] = defaultdict(list)
    for pattern in log_globs:
        # glob.glob, not Path.glob: the latter rejects absolute patterns, and
        # these are normally absolute (/e/project1/.../logs/slurm-*.log).
        for name in sorted(glob.glob(os.path.expanduser(pattern), recursive=True)):
            path = Path(name)
            if not path.is_file():
                continue
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            for label, regex, max_nodes in DIRECT_PATTERNS:
                # Collect this log's hits for this pattern FIRST, so the cap can
                # be applied to the whole file rather than to the first N lines.
                per_log: dict[str, str] = {}
                for match in regex.finditer(text):
                    node = match.group(1)
                    detail = (match.group(2) or "").strip()
                    entry = f"{label}: {detail}" if detail else label
                    per_log.setdefault(node, f"{entry} ({path.name})")
                if max_nodes and len(per_log) > max_nodes:
                    print(
                        f"note: {path.name}: '{label}' named {len(per_log)} distinct nodes "
                        f"(> {max_nodes}) — a mass crash, not an attribution; ignoring.",
                        file=sys.stderr,
                    )
                    continue
                for node, entry in per_log.items():
                    if entry not in hits[node]:
                        hits[node].append(entry)
    return hits


def _parse_slurm_time(value: str) -> datetime | None:
    value = (value or "").strip()
    if not value or value in {"Unknown", "None", "N/A"}:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


def collect_node_events(
    since: str, until: str | None
) -> dict[str, list[tuple[datetime, str, str]]]:
    """Historical node drain/down events: {node: [(when, state, reason), ...]}.

    `sacctmgr show event` is the accounting table of every node state change,
    and it is a strictly better instrument than `sinfo -R` for this job:
    `sinfo` only knows TODAY, so a node that broke during last week's run and
    was resumed the next morning shows nothing, while this table still has the
    event with its timestamp. It is also what makes a MID-RUN DEATH
    attributable at all — see attribute_mid_run_deaths().
    """
    cmd = [
        "sacctmgr",
        "-n",
        "-P",
        "show",
        "event",
        f"start={since}",
        "format=NodeName,TimeStart,TimeEnd,State,Reason",
    ]
    if until:
        cmd.insert(6, f"end={until}")
    rc, out = run(cmd, timeout=300)
    if rc != 0:
        print(
            "warning: `sacctmgr show event` unavailable — mid-run node failures cannot be "
            "attributed by drain timestamp.",
            file=sys.stderr,
        )
        return {}
    events: dict[str, list[tuple[datetime, str, str]]] = defaultdict(list)
    for line in out.splitlines():
        cols = line.split("|")
        if len(cols) < 5:
            continue
        node, t_start, _t_end, state, reason = cols[0], cols[1], cols[2], cols[3], cols[4]
        node = node.strip()
        # The table also carries cluster-level rows with an empty NodeName.
        if not node:
            continue
        when = _parse_slurm_time(t_start)
        if when is None:
            continue
        if not any(flag in state.upper() for flag in ("DRAIN", "DOWN", "FAIL")):
            continue
        if ADMIN_REASON_RE.search(reason):
            continue
        events[node].append((when, state.strip(), reason.strip()))
    return dict(events)


def attribute_mid_run_deaths(
    attempts: list[Attempt],
    events: dict[str, list[tuple[datetime, str, str]]],
    window_s: float,
    max_nodes: int,
) -> dict[str, list[str]]:
    """Attribute a job that DIED MIDWAY to the node that broke under it.

    THE OBSERVATION THIS RESTS ON. When SLURM kills an allocation because a node
    died, the node's drain event and the job's End timestamp are the SAME SECOND.
    Verified on all three of the campaign's node-failure kills:

        job 1375720 End 2026-08-14T22:07:17 | jpbo-060-24 DOWN* 22:07:17
        job 1395206 End 2026-08-17T12:39:53 | jpbo-026-42 DOWN* 12:39:53
        job 1400371 End 2026-08-18T05:21:34 | jpbo-107-45 DOWN* 05:21:34

    all three of which are entries someone previously identified BY HAND from
    the logs. This route needs no log at all, which is the point: it works when
    the monitor was not running, when the log was truncated or lost, and when
    the job simply hung and died without ever printing an error.

    A node qualifies when it was IN a failing allocation and a health drain/down
    event began inside [job start, job end + window]. `window_s` covers NHC
    running in the epilog rather than at the moment of death. Events before the
    job started are not considered: SLURM will not schedule onto an already
    drained node, so those describe a different incident.

    THE FALSE-POSITIVE CLASS THIS HAS TO REJECT is the job's own AFTERMATH.
    Replaying job 1395206, the two real culprits appear at +0 s — the node dying
    is what ended the job — but ten MORE nodes were drained 190-217 s later with

        NHC: check_stale_file_handles: Stale state file handle detected in: /e/project1

    which is the epilog finding the wreckage the dying job left on whichever
    nodes it happened to hold. Those nodes are fine. The tell is the COUNT: a
    fault attributes to one node, fallout attributes to many at once with the
    same reason, so hits are dropped when one job produces the same drain reason
    on more than `max_nodes` nodes. Same shape as the srun task-exit cap in
    DIRECT_PATTERNS, for the same underlying reason.
    """
    hits: dict[str, list[str]] = defaultdict(list)
    for attempt in attempts:
        if attempt.verdict != "fail":
            continue
        t0 = _parse_slurm_time(attempt.start)
        t1 = _parse_slurm_time(attempt.end)
        if t0 is None or t1 is None:
            continue
        # Group this job's in-window events by reason before accepting any, so
        # the cap sees the whole allocation rather than the first few nodes.
        by_reason: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for node in attempt.nodes:
            for when, state, reason in events.get(node, []):
                if when < t0 or (when - t1).total_seconds() > window_s:
                    continue
                delta = (when - t1).total_seconds()
                entry = (
                    f"drained {state} during job {attempt.job_id} "
                    f"({when:%Y-%m-%dT%H:%M:%S}, {delta:+.0f}s vs job end): {reason[:90]}"
                )
                by_reason[reason[:60]].append((node, entry))
        for reason_key, found in by_reason.items():
            nodes_hit = {node for node, _ in found}
            if len(nodes_hit) > max_nodes:
                print(
                    f"note: job {attempt.job_id}: {len(nodes_hit)} nodes drained with the same "
                    f"reason ({reason_key.strip()[:60]}...) — the job's own aftermath, not an "
                    f"attribution; ignoring.",
                    file=sys.stderr,
                )
                continue
            for node, entry in found:
                if entry not in hits[node]:
                    hits[node].append(entry)
    return hits


def collect_node_states(nodes: list[str]) -> dict[str, tuple[str, str]]:
    """Current SLURM state + drain reason for the given nodes.

    This is the corroboration step, and it is the one that turns a statistical
    suspicion into something worth writing down: every entry in the existing
    exclusion file that has held up was one where sinfo independently reported
    an NHC reason (check_hw_physmem_free, check_gpu_remapped_rows_pending, ...).
    """
    if not nodes:
        return {}
    rc, out = run(["sinfo", "-h", "-N", "-o", "%n|%T|%E"], timeout=120)
    if rc != 0:
        return {}
    wanted = set(nodes)
    states: dict[str, tuple[str, str]] = {}
    for line in out.splitlines():
        cols = line.split("|")
        if len(cols) < 3:
            continue
        name, state, reason = cols[0].strip(), cols[1].strip(), cols[2].strip()
        if name in wanted:
            states[name] = (state, reason)
    return states


def fleet_size(partition: str) -> int:
    rc, out = run(["sinfo", "-h", "-N", "-o", "%n", "-p", partition], timeout=120)
    if rc != 0:
        return 0
    return len({line.strip() for line in out.splitlines() if line.strip()})


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def analyze(args: argparse.Namespace) -> tuple[dict[str, NodeEvidence], dict[str, object]]:
    attempts = collect_attempts(args)
    direct_hits = collect_direct_hits(args.log_glob) if args.log_glob else {}

    fleet = args.fleet_size or fleet_size(args.partition)
    evidence: dict[str, NodeEvidence] = defaultdict(NodeEvidence)

    failing = [a for a in attempts if a.verdict == "fail"]
    for attempt in attempts:
        informative = attempt.verdict in ("fail", "ok")
        for node in attempt.nodes:
            evidence[node].appearances += 1
            if informative:
                evidence[node].informative += 1
    for attempt in failing:
        # Null expectation: if the bad node were random, each failing draw of
        # `nnodes` out of `fleet` would hit any given node with probability
        # nnodes/fleet. Summing over failing draws gives the appearances chance
        # alone explains — the ~6.5 figure the 2026-08-14 correlation used.
        share = (attempt.nnodes / fleet) if fleet else 0.0
        for node in attempt.nodes:
            evidence[node].failures += 1
            evidence[node].expected_failures += share
            if attempt.job_id not in evidence[node].fail_jobs:
                evidence[node].fail_jobs.append(attempt.job_id)

    # ---------------------------------------------------------------------
    # Significance. TWO tests, and only one of them is trustworthy.
    #
    # (a) APPEARANCE COUNT — "did this node show up in failing draws more often
    #     than chance?". This is the shape the manual 2026-08-14 correlation
    #     used, and on real data it is DOMINATED BY SLURM'S NODE ORDERING, not
    #     by node health: replaying those 35 attempts, entire racks (jpbo-036-*,
    #     jpbo-030-*) appear in 22-23 of 25 draws against an expectation of 4.2,
    #     at p=1e-7, because SLURM keeps handing out the same low-weight nodes.
    #     232 nodes cleared the bar. Kept as a diagnostic ranking, never acted on.
    #
    # (b) CONDITIONAL FAILURE RATE — "of the draws this node was IN, did more
    #     than its share fail?". A node SLURM always picks is in the successful
    #     draws too, so re-use cancels out; a node that actually breaks jobs is
    #     in failing draws and absent from successful ones. This is the test.
    #
    # (b) has NO POWER when every draw failed (q = 1): with nothing successful
    # to contrast against, counting cannot separate a bad node from bad luck,
    # and the honest answer is "this campaign cannot tell you" — which is
    # exactly the 2026-08-14 situation, where the real discriminator was the
    # independent NHC drain record, not the arithmetic.
    n_failing = len(failing)
    n_informative = sum(1 for a in attempts if a.verdict in ("fail", "ok"))
    q = (n_failing / n_informative) if n_informative else 0.0
    tests = max(fleet, len(evidence), 1)
    if fleet > 0 and n_failing > 0:
        p_draw = sum(a.nnodes for a in failing) / (n_failing * fleet)
        for ev in evidence.values():
            if ev.failures:
                ev.p_appearance = min(1.0, binomial_tail(ev.failures, n_failing, p_draw) * tests)
    elif n_failing > 0:
        print(
            "warning: fleet size unknown (sinfo unavailable?) — the appearance-count "
            "diagnostic is disabled; pass --fleet-size to enable it.",
            file=sys.stderr,
        )
    for ev in evidence.values():
        if ev.failures and ev.informative:
            ev.p_raw = binomial_tail(ev.failures, ev.informative, q)
            ev.p_corrected = min(1.0, ev.p_raw * tests)

    for node, entries in direct_hits.items():
        evidence[node].direct.extend(entries)

    # Mid-run deaths: correlate the failing allocations against SLURM's own
    # node-event table. Window derived from the attempts themselves rather than
    # from --since, so the sacctmgr query is bounded and cannot depend on the
    # user having typed a date format sacctmgr also accepts.
    starts = [t for t in (_parse_slurm_time(a.start) for a in attempts) if t]
    ends = [t for t in (_parse_slurm_time(a.end) for a in attempts) if t]
    timed_hits: dict[str, list[str]] = {}
    if starts and failing:
        window_start = min(starts).strftime("%Y-%m-%dT%H:%M:%S")
        # Padded past the last job end by the acceptance window: sacctmgr's
        # `end=` drops an event starting exactly on the boundary, and the
        # boundary is precisely where the interesting ones sit — the drain of
        # jpbo-107-45 is timestamped the same second job 1400371 ended, so an
        # unpadded query silently lost the campaign's clearest attribution.
        last = (max(ends) if ends else max(starts)) + timedelta(seconds=args.drain_window_s + 60)
        window_end = last.strftime("%Y-%m-%dT%H:%M:%S")
        events = collect_node_events(window_start, window_end)
        timed_hits = attribute_mid_run_deaths(
            attempts, events, args.drain_window_s, args.drain_max_nodes
        )
        for node, entries in timed_hits.items():
            evidence[node].timed.extend(entries)

    states = collect_node_states(list(evidence))
    for node, (state, reason) in states.items():
        evidence[node].slurm_state = state
        evidence[node].slurm_reason = reason

    # RACK GUARD. Count how many drain reasons repeat across the nodes that are
    # ELIGIBLE TO BE WRITTEN — i.e. those with >= --min-fails failing draws — and
    # let corroborated() refuse the ones shared too widely.
    #
    # THE POPULATION IS THE POINT, and getting it wrong makes the guard useless
    # in the other direction. Counting over every node in `evidence` (all 5428
    # that appeared in any draw) folds in the whole fleet's current drain state:
    # measured, that put 523 nodes under "Node not responding for N seconds" and
    # 19 under "Kill task failed", so the guard suppressed the four genuine
    # "Kill task failed" singles along with the racks. The question the guard
    # asks is narrower — "among the nodes I am about to write, is this reason
    # discriminating or is it one event repeated?" — so only those are counted.
    reason_counts: Counter[str] = Counter()
    for ev in evidence.values():
        if ev.failures < args.min_fails:
            continue
        if not ev.slurm_reason:
            continue
        if ev.slurm_reason.strip().lower() in {"none", "(null)"}:
            continue
        if ADMIN_REASON_RE.search(ev.slurm_reason):
            continue
        reason_counts[reason_key(ev.slurm_reason)] += 1
    for ev in evidence.values():
        ev.reason_max_nodes = args.corroborate_max_nodes
        if ev.slurm_reason:
            ev.reason_shared_by = reason_counts.get(reason_key(ev.slurm_reason), 0)
    suppressed = sorted(
        (
            (key, count)
            for key, count in reason_counts.items()
            if args.corroborate_max_nodes and count > args.corroborate_max_nodes
        ),
        key=lambda kv: -kv[1],
    )

    summary = {
        "attempts": len(attempts),
        "failing_attempts": len(failing),
        # Appearances among the failing draws that chance alone explains for any
        # one node — the "~6.5 out of 35" figure of the manual correlation.
        "expected_appearances": (
            round(sum(a.nnodes for a in failing) / fleet, 2) if fleet else 0.0
        ),
        "ok_attempts": sum(1 for a in attempts if a.verdict == "ok"),
        "neutral_attempts": sum(1 for a in attempts if a.verdict == "neutral"),
        "fleet_size": fleet,
        "nodes_seen": len(evidence),
        "direct_hits": len(direct_hits),
        "timed_hits": len(timed_hits),
        "failure_rate": round(q, 3),
        "job_ids": sorted({a.job_id for a in attempts}),
        "suppressed_reasons": suppressed,
    }
    return dict(evidence), summary


def select_candidates(
    evidence: dict[str, NodeEvidence], args: argparse.Namespace
) -> tuple[list[str], list[str], list[str]]:
    """Split the evidence into (direct, timed, statistical) candidate lists.

    The tiers are ordered by how much interpretation each needs, and a
    node is reported in the strongest tier it reaches.
    """
    direct = sorted(
        (n for n, e in evidence.items() if e.direct),
        key=lambda n: (-len(evidence[n].direct), n),
    )
    # A drain event inside the job window is evidence in its own right and needs
    # no failure count behind it: one such event on one failing job IS the
    # attribution. Requiring --min-fails here would discard exactly the case
    # this tier exists for — a run that died once, midway, on one bad node.
    timed = sorted(
        (n for n, e in evidence.items() if e.timed and not e.direct),
        key=lambda n: (-len(evidence[n].timed), n),
    )

    # TWO ROUTES IN, WITH DELIBERATELY ASYMMETRIC BARS.
    #
    # (a) CORROBORATED — the node was in >= --min-fails of our failing draws AND
    #     sinfo independently reports a HEALTH drain reason for it. No
    #     significance bar. That is not sloppiness, it is the cost structure:
    #     excluding a node costs 1/5565 of the fleet, missing one costs a
    #     512-node job, and the drain reason is evidence from a completely
    #     different instrument (NHC) rather than another slice of the same
    #     counts. Requiring family-wise significance ON TOP of it over-corrects
    #     to the point of uselessness — measured on the 2026-08-15..20 window,
    #     the top eight suspects ALL had hardware NHC reasons (uncorrected GPU
    #     SRAM errors, link-error counters, unresponsive twins) and not one
    #     cleared a Bonferroni bar over 5565 nodes.
    #
    # (b) UNCORROBORATED — nothing but our own counts, so the bar is the strict
    #     fleet-corrected conditional test. Hard to clear on purpose: with no
    #     second instrument, this is the only thing standing between the file
    #     and a rack that SLURM merely happens to schedule first.
    def qualifies(ev: NodeEvidence) -> bool:
        if ev.failures < args.min_fails:
            return False
        if ev.corroborated:
            return True
        return ev.p_corrected <= args.alpha or (
            args.include_uncorroborated and ev.p_raw <= args.alpha
        )

    statistical = sorted(
        (n for n, e in evidence.items() if not e.direct and not e.timed and qualifies(e)),
        key=lambda n: (
            0 if evidence[n].corroborated else 1,
            evidence[n].p_raw,
            evidence[n].p_appearance,
            n,
        ),
    )
    return direct, timed, statistical


def print_report(
    evidence: dict[str, NodeEvidence],
    summary: dict[str, object],
    direct: list[str],
    timed: list[str],
    statistical: list[str],
    args: argparse.Namespace,
) -> None:
    print("=" * 78)
    print("NODE-CATCH ANALYSIS")
    print("=" * 78)
    print(
        f"attempts={summary['attempts']}  failing={summary['failing_attempts']}  "
        f"ok={summary['ok_attempts']}  neutral={summary['neutral_attempts']}  "
        f"fleet={summary['fleet_size']}  nodes_seen={summary['nodes_seen']}  "
        f"failure_rate={summary['failure_rate']}"
    )
    if summary["failing_attempts"] and not summary["ok_attempts"]:
        print(
            "\nNOTE: every informative draw failed, so the conditional test has no power —\n"
            "      with nothing successful to contrast against, counting cannot separate a\n"
            "      bad node from bad luck. Rely on DIRECT evidence and the sinfo drain\n"
            "      reasons below; the appearance ranking is shown for orientation only."
        )
    for key, count in summary.get("suppressed_reasons", []):
        print(
            f'\nRACK GUARD: {count} candidate nodes share the drain reason "{key.strip()}"\n'
            f"      (> {args.corroborate_max_nodes}) — one infrastructure event, not {count} "
            f"faults, so it does NOT\n      corroborate. Those nodes must clear the statistical "
            f"bar on their own counts.\n      Raise --corroborate-max-nodes to treat it as "
            f"evidence."
        )
    if not summary["attempts"]:
        print("\nNo allocations matched the selection — widen --since or check --name/--user.")
        return
    if not summary["failing_attempts"] and not direct and not timed:
        print("\nNo failing allocations and no direct hits. Nothing to exclude — good news.")

    def show(node: str) -> None:
        ev = evidence[node]
        state = f"{ev.slurm_state}" if ev.slurm_state else "?"
        print(
            f"  {node}  fails={ev.failures}/{ev.informative} draws  "
            f"expected={ev.expected_failures:.1f}  p={ev.p_raw:.2g} "
            f"(corr {ev.p_corrected:.2g})  p_appear={ev.p_appearance:.2g}  state={state}"
        )
        for entry in ev.direct[:4]:
            print(f"      evidence: {entry}")
        for entry in ev.timed[:4]:
            print(f"      drain:    {entry}")
        if ev.corroborated and ev.slurm_reason:
            print(f"      sinfo:    {ev.slurm_reason}")
        if ev.fail_jobs:
            print(f"      jobs:     {','.join(ev.fail_jobs[:8])}")

    if direct:
        print(f"\nDIRECT — named by a failure line or probe verdict ({len(direct)}):")
        for node in direct:
            show(node)

    if timed:
        print(
            f"\nDIED UNDER US — a health drain/down event began while this node was "
            f"running one of our failing jobs ({len(timed)}):"
        )
        for node in timed:
            show(node)

    shown_cap = args.max_report
    if statistical:
        corroborated = [n for n in statistical if evidence[n].corroborated]
        print(
            f"\nSTATISTICAL — >= {args.min_fails} failing draws plus either an independent "
            f"health drain reason, or p <= {args.alpha} fleet-corrected "
            f"({len(statistical)}, {len(corroborated)} corroborated by sinfo):"
        )
        for node in statistical[:shown_cap]:
            show(node)
        if len(statistical) > shown_cap:
            print(f"  ... {len(statistical) - shown_cap} more (raise --max-report to see them)")
    elif summary["failing_attempts"]:
        print(
            f"\nSTATISTICAL — nothing clears the bar "
            f"(>= {args.min_fails} failing draws, Bonferroni p <= {args.alpha}). "
            "With few draws this is the normal outcome: a handful of allocations "
            "simply cannot single a node out."
        )
        # Corroborated nodes first: an independent NHC drain reason beats any
        # amount of arithmetic, and it is the thing a human should look at.
        ranked = sorted(
            (kv for kv in evidence.items() if kv[1].failures),
            key=lambda kv: (
                0 if kv[1].corroborated else 1,
                kv[1].p_raw,
                kv[1].p_appearance,
                -kv[1].failures,
            ),
        )[:8]
        if ranked:
            print("  most suspicious anyway (ranking only — do NOT exclude on this alone):")
            for node, ev in ranked:
                mark = " [sinfo: " + ev.slurm_reason[:60] + "]" if ev.corroborated else ""
                print(
                    f"    {node}  fails={ev.failures}/{ev.informative}"
                    f"  expected={ev.expected_failures:.1f}  p={ev.p_raw:.2g}"
                    f"  p_appear={ev.p_appearance:.2g}{mark}"
                )


# ---------------------------------------------------------------------------
# exclusion-file I/O
# ---------------------------------------------------------------------------
def read_exclude_entries(path: Path) -> list[str]:
    """Node names currently active in the file (mirrors the oc.exclude_nodes
    resolver).

    Only a leading '#' comments a line out — the resolver does NOT strip
    trailing comments, which is why the file's own header forbids them.
    """
    if not path.exists():
        return []
    nodes: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        nodes.extend(token for token in re.split(r"[,\s]+", line) if token)
    return nodes


def render_block(
    evidence: dict[str, NodeEvidence],
    summary: dict[str, object],
    direct: list[str],
    timed: list[str],
    statistical: list[str],
    already: set[str],
    args: argparse.Namespace,
) -> tuple[str, list[str]]:
    """Render the annotated block to append, plus the nodes it adds."""
    today = datetime.now().strftime("%Y-%m-%d")
    new_direct = [n for n in direct if n not in already]
    new_timed = [n for n in timed if n not in already]
    # select_candidates() already applied the two-tier bar (corroborated, or
    # significant on our own counts alone), so everything here is writable. The
    # split is preserved in the comments so a later `prune` can tell which
    # entries rest on a second instrument and which rest only on arithmetic.
    new_stat = [n for n in statistical if n not in already]
    added = new_direct + new_timed + new_stat
    if not added:
        return "", []

    lines = [
        "",
        "# " + "-" * 75,
        f"# {today} — scripts/run_autoexclude_jupiter.py {args.command}",
        "# " + "-" * 75,
        f"# Selection: {'jobs ' + args.jobs if args.jobs else f'user {args.user} since {args.since}'}"
        + (f", name~{args.name}" if args.name else ""),
        f"# {summary['attempts']} allocation attempts, {summary['failing_attempts']} failing, "
        f"fleet {summary['fleet_size']} nodes.",
        f"# Campaign failure rate {summary['failure_rate']}; a random node is expected in "
        f"~{summary['expected_appearances']} of the failing draws. Statistical entries below "
        f"clear a Bonferroni-corrected conditional-failure p <= {args.alpha}",
        "# OR carry an independent NHC/health drain reason from sinfo.",
    ]
    if args.note:
        lines.append(f"# NOTE: {args.note}")

    if new_direct:
        lines += ["#", "# DIRECT — a failure line or probe verdict named this node."]
        for node in new_direct:
            ev = evidence[node]
            for entry in ev.direct[:3]:
                lines.append(f"#   {node}  {entry}")
            # ev.corroborated, not ev.slurm_reason: sinfo prints the literal
            # string "none" for an empty reason, so the bare truthiness test
            # writes `sinfo: none` into the file as if it were evidence.
            if ev.corroborated:
                lines.append(f"#   {node}  sinfo: {ev.slurm_reason}")
        lines += new_direct

    if new_timed:
        lines += [
            "#",
            "# DIED UNDER US — SLURM drained/downed this node while it was running one of",
            "# the failing jobs below. No log line needed; the timestamps are the evidence.",
        ]
        for node in new_timed:
            for entry in evidence[node].timed[:3]:
                lines.append(f"#   {node}  {entry}")
        lines += new_timed

    if new_stat:
        lines += [
            "#",
            "# STATISTICAL — this node was in several failing allocations. Weaker evidence",
            "# than a named failure; prune these first if the fleet gets tight. Entries with",
            "# a sinfo reason rest on an independent instrument (NHC); those without rest",
            "# only on our own counts and cleared the fleet-corrected bar.",
        ]
        for node in new_stat:
            ev = evidence[node]
            detail = (
                f"#   {node}  failed {ev.failures} of its {ev.informative} draws "
                f"(p={ev.p_raw:.2g}, fleet-corrected {ev.p_corrected:.2g})"
            )
            if ev.slurm_reason:
                detail += f" | sinfo: {ev.slurm_reason}"
                # Say so when the reason did NOT corroborate, or a later reader
                # takes the sinfo string as the second instrument it is not.
                if not ev.corroborated:
                    detail += (
                        f" [shared by {ev.reason_shared_by} nodes — rack event, "
                        f"did not corroborate]"
                    )
            lines.append(detail)
        lines += new_stat

    return "\n".join(lines) + "\n", added


def cmd_update(args: argparse.Namespace) -> int:
    evidence, summary = analyze(args)
    direct, timed, statistical = select_candidates(evidence, args)
    print_report(evidence, summary, direct, timed, statistical, args)

    path = Path(args.exclude_file)
    already = set(read_exclude_entries(path))
    block, added = render_block(evidence, summary, direct, timed, statistical, already, args)

    print("\n" + "=" * 78)
    if not added:
        skipped = [n for n in direct + timed + statistical if n in already]
        if skipped:
            print(f"Nothing to add — all {len(skipped)} candidate(s) are already in {path.name}.")
        else:
            print(f"Nothing to add to {path}.")
        return 0

    print(f"Would add {len(added)} node(s) to {path}:\n")
    print(block)
    if not args.apply:
        print("DRY RUN — re-run with --apply to write. The file is git-tracked; review the diff.")
        return 0

    with path.open("a", encoding="utf-8") as handle:
        handle.write(block)
    print(f"WROTE {len(added)} node(s) to {path}")
    print("Next submission picks them up automatically (oc.exclude_nodes re-reads on every")
    print("render). Jobs already PENDING do not — the monitor's UpdateChainExcludesAction")
    print("handles those, or push by hand:  scontrol update JobId=<id> ExcNodeList=<list>")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    evidence, summary = analyze(args)
    direct, timed, statistical = select_candidates(evidence, args)
    print_report(evidence, summary, direct, timed, statistical, args)
    if args.json:
        payload = {
            "summary": summary,
            "direct": direct,
            "timed": timed,
            "statistical": statistical,
            "nodes": {
                node: {
                    "appearances": ev.appearances,
                    # The denominator the report prints as fails=X/Y. Without it
                    # a JSON dump cannot reproduce the conditional test.
                    "informative": ev.informative,
                    "failures": ev.failures,
                    "expected_failures": round(ev.expected_failures, 3),
                    "ratio": round(ev.ratio, 3),
                    "p_raw": ev.p_raw,
                    "p_corrected": ev.p_corrected,
                    "corroborated": ev.corroborated,
                    "reason_shared_by": ev.reason_shared_by,
                    "direct": ev.direct,
                    "timed": ev.timed,
                    "fail_jobs": ev.fail_jobs,
                    "slurm_state": ev.slurm_state,
                    "slurm_reason": ev.slurm_reason,
                }
                for node, ev in sorted(evidence.items())
                if ev.failures or ev.direct or ev.timed
            },
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    """Comment out entries whose node is healthy again.

    SLURM already refuses to schedule onto a DRAINED node, so an entry
    only does real work in the window after an admin resumes a node
    whose root cause was never fixed. Once the node is back in service
    and passing NHC, keeping it excluded just shrinks the pool a
    512-node job can schedule on.
    """
    path = Path(args.exclude_file)
    entries = read_exclude_entries(path)
    if not entries:
        print(f"{path} holds no active entries.")
        return 0
    states = collect_node_states(entries)
    if not states:
        print("error: could not read node states from sinfo — run this on a login node.")
        return 1

    healthy, still_bad, unknown = [], [], []
    for node in entries:
        state, reason = states.get(node, ("", ""))
        if not state:
            unknown.append(node)
        elif any(flag in state.lower() for flag in ("drain", "down", "fail", "maint", "unk")):
            still_bad.append((node, state, reason))
        else:
            healthy.append((node, state, reason))

    print("=" * 78)
    print(f"PRUNE CHECK — {path}")
    print("=" * 78)
    print(
        f"{len(entries)} active entries: {len(healthy)} healthy, {len(still_bad)} still bad, "
        f"{len(unknown)} unknown"
    )
    if still_bad:
        print("\nSTILL BAD (keep):")
        for node, state, reason in still_bad:
            print(f"  {node}  {state}  {reason}")
    if unknown:
        print("\nUNKNOWN to sinfo (keep — probably removed from the fleet):")
        for node in unknown:
            print(f"  {node}")
    if not healthy:
        print("\nNothing to prune.")
        return 0

    print("\nHEALTHY AGAIN (prunable):")
    for node, state, reason in healthy:
        print(f"  {node}  {state}  {reason or '(no reason recorded)'}")
    print(
        "\nNB 'healthy' here means SLURM will schedule it. It does NOT mean the original\n"
        "root cause was fixed — that is exactly the window this file exists to cover.\n"
        "Prune the STATISTICAL entries first, and keep anything whose comment records a\n"
        "hardware fault (row remapping, ECC, GSP) until the node has been reset."
    )
    if not args.apply:
        print("\nDRY RUN — re-run with --apply to comment these out.")
        return 0

    today = datetime.now().strftime("%Y-%m-%d")
    prune_set = {node for node, _, _ in healthy}
    out_lines: list[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and stripped in prune_set:
            out_lines.append(f"# [pruned {today}: node healthy again] {stripped}")
        else:
            out_lines.append(line)
    path.write_text("\n".join(out_lines) + "\n")
    print(f"\nCommented out {len(prune_set)} entr(ies) in {path} — review the git diff.")
    return 0


def cmd_campaign(args: argparse.Namespace) -> int:
    """Run the node-catching experiment, then curate from its own jobs.

    THIS NEEDS A TTY, OR A "y" ON STDIN. run_autoexp.py gates any plan over
    100 GPU-h behind an interactive `Proceed? [y/N]` (run_autoexp.py:308) and a
    512-node/20-min draw is ~683, so the gate always fires here. It reads stdin,
    which this subprocess inherits, and treats EOF as "no". Launched detached
    with `< /dev/null` the campaign therefore prints "launching", submits
    NOTHING, and then curates an empty window and reports success — which looks
    exactly like a clean run that found no bad nodes. Feed it instead:

        printf 'y\\ny\\n' | setsid nohup python scripts/run_autoexclude_jupiter.py \\
            campaign --config-name experiments/oellm_32b_dense/node_catch_n512 \\
            --apply > dump/nodecatch.log 2>&1 &

    TWO LINES, NOT ONE, FOR THE 512-NODE CAMPAIGN. run_autoexp asks a SECOND
    `Really proceed?` above 5000 GPU-h, and node_catch_n512 is 5,461 (8 draws x
    512 nodes x 4 GPUs x 20 min) now that the estimator prices chain repeats —
    it used to quote 683, i.e. one draw, and slipped under the second gate. So:

        printf 'y\\ny\\n' | setsid nohup python ... &
    """
    started = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_autoexp.py"),
        "--config-name",
        args.config_name,
        *args.passthrough,
    ]
    env = dict(os.environ)
    # PREPEND, do not setdefault. A JUPITER login shell already exports a
    # PYTHONPATH (the Lmod/uv modules put their site-packages there), so
    # setdefault is a silent no-op exactly where this runs, and run_autoexp.py
    # dies with `ModuleNotFoundError: No module named 'oellm_autoexp'` — after
    # the campaign has printed "launching", and before it has submitted
    # anything, so the run then curates an empty window and reports success.
    # Measured 2026-08-23 (PYTHONPATH=.../uv/0.8.17-GCCcore-14.3.0/...).
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
    print("=" * 78)
    print("CAMPAIGN — launching:", " ".join(cmd))
    print(f"(analysis window opens at {started})")
    print("=" * 78)
    rc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False).returncode
    print(f"\nrun_autoexp exited rc={rc}; curating allocations since {started}\n")

    args.since = started
    args.until = None
    args.jobs = ""
    args.command = "campaign"
    return cmd_update(args)


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_selection(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--jobs", default="", help="Explicit SLURM job ids (comma separated).")
        sp.add_argument("--since", default="today", help="sacct -S window start (default: today).")
        sp.add_argument("--until", default=None, help="sacct -E window end.")
        sp.add_argument("--user", default=os.environ.get("USER", ""), help="sacct -u user.")
        sp.add_argument("--name", default="", help="Only jobs whose JobName contains this.")
        sp.add_argument(
            "--min-nodes",
            type=int,
            default=2,
            help="Ignore allocations smaller than this (default 2: single-node jobs carry "
            "almost no attribution power and swamp the tallies).",
        )
        sp.add_argument(
            "--cancelled-as-fail",
            action="store_true",
            help="Count CANCELLED allocations as failures (default: neutral — an operator "
            "scancel and a monitor CancelAction are indistinguishable in sacct).",
        )
        sp.add_argument(
            "--log-glob",
            action="append",
            default=[],
            help="Also harvest node-naming failure lines from these logs (repeatable). "
            "Needed when no monitor was running to record them live.",
        )
        sp.add_argument("--partition", default="booster", help="Partition used for fleet size.")
        sp.add_argument(
            "--fleet-size", type=int, default=0, help="Override the fleet size used as the null."
        )
        sp.add_argument(
            "--min-fails",
            type=int,
            default=2,
            help="Statistical candidates need at least this many failing draws (default 2).",
        )
        sp.add_argument(
            "--alpha",
            type=float,
            default=0.05,
            help="...and a Bonferroni-corrected conditional-failure-rate p at or below this "
            "(default 0.05). Neither a ratio bar nor an appearance-count test works here — "
            "see the module docstring for what each of them returned on real data.",
        )
        sp.add_argument(
            "--include-uncorroborated",
            action="store_true",
            help="Lower the bar for candidates with NO corroborating sinfo reason from the "
            "fleet-corrected p to the raw p. Off by default: draws are not independent "
            "samples, so uncorroborated hits can be an artefact of SLURM node ordering.",
        )
        sp.add_argument(
            "--drain-window-s",
            type=float,
            default=900.0,
            help="A node counts as having died under a job when a health drain/down event "
            "began between the job's start and its end + this many seconds (default 900). "
            "The slack covers NHC running in the epilog rather than at the moment of death; "
            "on the three observed node-failure kills the event and the job End matched to "
            "the second.",
        )
        sp.add_argument(
            "--drain-max-nodes",
            type=int,
            default=3,
            help="If one job drains more than this many nodes with the same reason, the "
            "drains are treated as the job's own aftermath (stale file handles left by a "
            "dying job, etc.) and ignored rather than attributed (default 3).",
        )
        sp.add_argument(
            "--corroborate-max-nodes",
            type=int,
            default=8,
            help="If more than this many of the nodes seen share ONE sinfo drain reason, that "
            "reason is an infrastructure/rack event rather than a per-node fault and stops "
            "corroborating (default 8; 0 disables the guard). Measured 2026-08-23: without it, "
            "80 of 84 candidates were four whole cells reading the same 'Node not responding' "
            "string. Such nodes can still be written — they just have to clear the statistical "
            "bar on their own counts.",
        )
        sp.add_argument(
            "--max-report",
            type=int,
            default=25,
            help="Cap on statistical candidates printed (the total is always reported).",
        )

    p_analyze = sub.add_parser("analyze", help="Report suspect nodes (read-only).")
    add_selection(p_analyze)
    p_analyze.add_argument("--json", default="", help="Also write the full result as JSON.")
    p_analyze.set_defaults(func=cmd_analyze)

    p_update = sub.add_parser("update", help="Append suspects to the exclusion file.")
    add_selection(p_update)
    p_update.add_argument("--exclude-file", default=str(DEFAULT_EXCLUDE_FILE))
    p_update.add_argument("--note", default="", help="Free-text note recorded in the block.")
    p_update.add_argument("--apply", action="store_true", help="Actually write (default: dry run).")
    p_update.set_defaults(func=cmd_update)

    p_prune = sub.add_parser("prune", help="Comment out entries whose node is healthy again.")
    p_prune.add_argument("--exclude-file", default=str(DEFAULT_EXCLUDE_FILE))
    p_prune.add_argument("--apply", action="store_true", help="Actually write (default: dry run).")
    p_prune.set_defaults(func=cmd_prune)

    p_camp = sub.add_parser("campaign", help="Run an experiment, then curate from its jobs.")
    p_camp.add_argument(
        "--config-name",
        default="experiments/oellm_32b_dense/node_catch_n512",
        help="Experiment config passed to run_autoexp.py.",
    )
    add_selection(p_camp)
    p_camp.add_argument("--exclude-file", default=str(DEFAULT_EXCLUDE_FILE))
    p_camp.add_argument("--note", default="", help="Free-text note recorded in the block.")
    p_camp.add_argument("--apply", action="store_true", help="Actually write (default: dry run).")
    p_camp.add_argument(
        "passthrough",
        nargs="*",
        default=[],
        help="Extra args forwarded verbatim to run_autoexp.py (e.g. job.chain_repeat=4).",
    )
    p_camp.set_defaults(func=cmd_campaign)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
