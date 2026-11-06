#!/usr/bin/env python3
"""Generate the GROUPED export -> eval pipeline config: the 438 FLAG rows in a few groups that
each load the model once (mn5_oellm_run_one.sh with comma-separated task / n_shot lists).

One row per job loads the 84B export per row: 4-14 min each on MareNostrum 5 (21 min on a
slow node), ~438 loads per checkpoint, while many rows run 1-2 min. Here a group = one job =
one load, the rows of one harness (lm-eval / lighteval / evalchemy) balanced by expected time
(longest first onto the least loaded group): the times of a reference run where known (the
row's measured job time minus the load), else the harness median. The nodes are split over the
harnesses in proportion to their total time (at least one each).

It inherits flag_eval_pipeline_mn5 (export stage, aux.step / aux.model knobs, eval env) and
replaces only the sweep: point 0 the export stage, points 1..G the groups, gated on the
export's .downstream_ready.

    python scripts/korbi/flag_eval/gen_grouped_config.py --nodes 16 \\
        > config/experiments/korbi/production/flag_eval_grouped_mn5.yaml

    scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py --no-monitor \\
        --monitor-state-dir $R/pool_state --pool <pool> \\
        --config-name experiments/korbi/production/flag_eval_grouped_mn5 \\
        aux.model=ambitious_v4_hd256 aux.step=10000 --yes
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import statistics
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
SUITE = REPO / "config/eval_tasks/flag_evals.yaml"
TIMES = REPO / "config/eval_tasks/flag_evals_row_times.json"
HARNESSES = ("lm-eval-harness", "lighteval", "evalchemy")
# lm-eval groups on the fast runner's layout (flag_eval_fast_mn5, experiment 008/012): TP1 x DP4
# x EP, request batches of 256, the Engram tables shared host memory (uva: one copy for the four
# engines), no decode graphs. The per-row default TP4 ran the same rows ~3x slower (11-06:
# mmlu/5 4342 s vs 1311 s). lighteval / evalchemy keep TP4 (one engine over the four GPUs).
LM_EVAL_ENV = {
    "OELLM_TP": "1", "OELLM_DP": "4", "OELLM_EP": "1", "OELLM_REQUEST_BATCH_SIZE": "256",
    "OELLM_ENGRAM_TABLES": "uva", "OELLM_DECODE_GRAPHS": "0",
}
# The task list in single quotes: run_tests.sh gets the bench as one double-quoted argument, so
# double quotes around ${aux.task} ended it and the list was split at a space in a task name
# ("include_base_44_north macedonian": '$4: unbound variable', 11-06).
BENCH = ("bash scripts/korbi/flag_eval/mn5_oellm_run_one.sh ${aux.export_dir} ${aux.out} "
         "'${aux.task}' ${aux.n_shot} ${aux.suite} ${aux.limit}")


def expected_seconds(rows: list[dict], times: dict[str, float], load_s: float) -> list[float]:
    """Per row: the reference job time minus the load, else the median of its harness."""
    key = lambda r: f"{r['suite']}|{r['task']}|{r['n_shot']}"  # noqa: E731
    known: dict[str, list[float]] = {}
    for r in rows:
        if key(r) in times:
            known.setdefault(r["suite"], []).append(max(60.0, times[key(r)] - load_s))
    med = {s: statistics.median(v) for s, v in known.items()}
    fallback = statistics.median([x for v in known.values() for x in v]) if known else 600.0
    return [
        max(60.0, times[key(r)] - load_s) if key(r) in times else med.get(r["suite"], fallback)
        for r in rows
    ]


def split_nodes(total: dict[str, float], nodes: int) -> dict[str, int]:
    """Groups per harness, in proportion to its total time, at least one each (largest
    remainders get the leftover nodes)."""
    present = [h for h in HARNESSES if total.get(h, 0) > 0]
    if nodes < len(present):
        raise SystemExit(f"--nodes {nodes} < {len(present)} harnesses")
    grand = sum(total[h] for h in present)
    share = {h: max(1.0, nodes * total[h] / grand) for h in present}
    out = {h: max(1, math.floor(share[h])) for h in present}
    while sum(out.values()) > nodes:
        h = max((h for h in present if out[h] > 1), key=lambda h: out[h] - share[h])
        out[h] -= 1
    for h in sorted(present, key=lambda h: share[h] - out[h], reverse=True):
        if sum(out.values()) >= nodes:
            break
        out[h] += 1
    return out


def lpt(items: list[tuple[float, int]], groups: int) -> list[list[int]]:
    """Longest first onto the least loaded group; each group's rows in suite order."""
    heap = [(0.0, g) for g in range(groups)]
    out: list[list[int]] = [[] for _ in range(groups)]
    for secs, i in sorted(items, reverse=True):
        load, g = heapq.heappop(heap)
        out[g].append(i)
        heapq.heappush(heap, (load + secs, g))
    return [sorted(g) for g in out if g]


def build(rows: list[dict], times: dict[str, float], nodes: int, load_s: float) -> list[dict]:
    secs = expected_seconds(rows, times, load_s)
    total = {h: sum(s for s, r in zip(secs, rows) if r["suite"] == h) for h in HARNESSES}
    per = split_nodes(total, nodes)
    groups = []
    for h in HARNESSES:
        idx = [(secs[i], i) for i, r in enumerate(rows) if r["suite"] == h]
        for members in lpt(idx, per.get(h, 0)) if idx else []:
            # evalchemy runs one harness call (one load) per task (mn5_oellm_run_one.sh)
            loads = len(members) if h == "evalchemy" else 1
            groups.append({
                "suite": h,
                "rows": members,
                "seconds": loads * load_s + sum(secs[i] for i in members),
            })
    return groups


def render(rows: list[dict], groups: list[dict], nodes: int, tag: str = "") -> str:
    out = [
        "# @package _global_",
        "# GENERATED by scripts/korbi/flag_eval/gen_grouped_config.py -- do not edit.",
        f"# {len(rows)} FLAG rows in {len(groups)} groups (one model load each) for {nodes} nodes,",
        "# + the export stage of flag_eval_pipeline_mn5 (aux.step / aux.model knobs).",
        "defaults:",
        "  - flag_eval_pipeline_mn5",
        "  - _self_",
        "",
        "aux:",
        f"  bench: {json.dumps(BENCH)}",
        "",
        "sweep:",
        "  type: list",
        "  store_sweep_json: true",
        "  groups:",
        "    - type: list",
        "      configs:",
        "        - aux.row: 0",
        '          aux.name: "export_${aux.export}"',
        '          aux.task: "export"',
        '          aux.suite: "pipeline"',
        "          aux.n_shot: 0",
        "          aux.bench: ${aux.export_bench}",
        "          job.est_step_time_s: 1800",
        '          slurm.sbatch.time: "00:30:00"',
        "          job.metadata.pool_priority: 50",
        "          backend.env.PYTHONPATH: \".:submodules/Megatron-LM:flash-linear-attention:"
        "submodules/cute_kernels:${aux.images.${aux.image}.libs}:${aux.hybrid_env}/pylibs_hfx\"",
    ]
    for g, grp in enumerate(groups, start=1):
        members = [rows[i] for i in grp["rows"]]
        # the wall clock: twice the expectation, at least 2 h, at most 2 days
        limit_h = min(48, max(2, math.ceil(2 * grp["seconds"] / 3600)))
        out += [
            f"        - aux.row: {g}",
            f'          aux.name: "oellm${{aux.step}}g{tag}_{g}_{grp["suite"]}_{len(members)}rows"',
            f"          aux.task: {json.dumps(','.join(r['task'] for r in members))}",
            f"          aux.n_shot: {json.dumps(','.join(str(int(r['n_shot'])) for r in members))}",
            f"          aux.suite: {json.dumps(grp['suite'])}",
            f"          job.est_step_time_s: {int(grp['seconds'])}",
            f'          slurm.sbatch.time: "{limit_h:02d}:00:00"',
            *(
                f"          backend.env.{k}: {json.dumps(v)}"
                for k, v in (LM_EVAL_ENV.items() if grp["suite"] == "lm-eval-harness" else ())
            ),
            "          job.start_condition:",
            "            class_name: FileExistsCondition",
            '            path: "${aux.export_dir}/.downstream_ready"',
            "            persistent_pass: true",
        ]
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nodes", type=int, default=16, help="groups = concurrent one-node jobs")
    p.add_argument("--load-s", type=float, default=600.0,
                   help="the model load inside a reference row time (subtracted)")
    p.add_argument("--times", type=Path, default=TIMES)
    p.add_argument("--only", type=Path,
                   help="JSON list of <suite>|<task>|<n_shot> keys: group only these rows (a rerun of "
                        "the rows without a result)")
    p.add_argument("--tag", default="", help="job-name tag, e.g. r1 for a rerun")
    a = p.parse_args()
    rows = yaml.safe_load(SUITE.read_text())["tasks"]
    if a.only:
        keep = set(json.loads(a.only.read_text()))
        rows = [r for r in rows if f"{r['suite']}|{r['task']}|{r['n_shot']}" in keep]
    times = json.loads(a.times.read_text()) if a.times.exists() else {}
    groups = build(rows, times, a.nodes, a.load_s)
    print(render(rows, groups, a.nodes, a.tag), end="")


if __name__ == "__main__":
    main()
