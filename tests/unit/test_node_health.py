"""oellm_autoexp/node_health: the verdict over benchmark results (synthetic and real MareNostrum 5
logs) and the benchmark itself, run on CPU / gloo with two local torchrun agents."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from oellm_autoexp.node_health.bench import PREFIX, summary
from oellm_autoexp.node_health.verdict import judge

DATA = Path(__file__).parent / "data" / "node_health"
REPO = Path(__file__).resolve().parents[2]


def _result(nodes=4, gpus=4, flops=None, groups=None, pairs=None) -> dict:
    names = [f"n{i}" for i in range(nodes)]
    return {
        "nodes": names,
        "gpus_per_node": gpus,
        "device": "cuda",
        "compute": flops or {n: [400.0] * gpus for n in names},
        "groups": groups
        or {str(g): {"all_gather": 22.0, "reduce_scatter": 22.5} for g in range(gpus)},
        "pairs": pairs
        if pairs is not None
        else [
            [names[(i // gpus - 1) % nodes], names[i // gpus], i % gpus, 12.5]
            for i in range(nodes * gpus)
        ],
        "sizes_mb": {"collective": 256, "ring": 512},
    }


def _log(result: dict) -> str:
    return "\n".join(["noise", *summary(result), PREFIX + json.dumps(result), "[Gloo] more noise"])


def test_clean_result_passes():
    v = judge(_log(_result()))
    assert v.status == "pass" and not v.exclude, v


def test_slow_gpu_excludes_its_node():
    r = _result()
    r["compute"]["n2"] = [400.0, 400.0, 120.0, 400.0]  # a throttled GPU
    v = judge(_log(r))
    assert (v.status, v.exclude) == ("exclude", ["n2"]), v
    assert "matmul" in v.why


def test_degraded_rail_excludes_the_node_common_to_the_two_slowest_pairs():
    """n2's GPU-2 NIC: both its incoming (n1->n2) and outgoing (n2->n3) pairs on rail 2 are slow;
    the single slowest pair is n1->n2, so 'slowest receiver' alone would wrongly also blame n1."""
    groups = {str(g): {"all_gather": 22.0, "reduce_scatter": 22.5} for g in range(4)}
    groups["2"] = {"all_gather": 11.9, "reduce_scatter": 11.8}
    r = _result(groups=groups)
    for p in r["pairs"]:
        if p[2] == 2 and (p[0], p[1]) == ("n1", "n2"):
            p[3] = 8.0
        if p[2] == 2 and (p[0], p[1]) == ("n2", "n3"):
            p[3] = 8.1
    v = judge(_log(r))
    assert (v.status, v.exclude) == ("exclude", ["n2"]), v


def test_no_common_node_excludes_both_ends_of_the_slowest_pair():
    groups = {str(g): {"all_gather": 22.0, "reduce_scatter": 22.5} for g in range(4)}
    groups["1"] = {"all_gather": 9.0, "reduce_scatter": 9.0}
    r = _result(groups=groups)
    for p in r["pairs"]:
        if p[2] == 1 and (p[0], p[1]) == ("n0", "n1"):
            p[3] = 5.0
        if p[2] == 1 and (p[0], p[1]) == ("n2", "n3"):
            p[3] = 6.0
    v = judge(_log(r))
    assert (v.status, v.exclude) == ("exclude", ["n0", "n1"]), v


def test_slow_group_without_ring_pairs_is_an_error():
    groups = {str(g): {"all_gather": 22.0, "reduce_scatter": 22.5} for g in range(4)}
    groups["3"] = {"all_gather": 5.0, "reduce_scatter": 5.0}
    v = judge(_log(_result(groups=groups, pairs=[])))
    assert v.status == "error", v


def test_no_result_is_an_error():
    assert judge("Traceback ...\nNCCL error\n").status == "error"


def test_single_node_passes_on_compute_alone():
    r = _result(nodes=1, groups={}, pairs=[])
    assert judge(_log(r)).status == "pass"


def test_the_last_result_in_a_log_counts():
    bad = _result()
    bad["compute"]["n0"] = [10.0, 400.0, 400.0, 400.0]
    assert judge(_log(bad) + "\n" + _log(_result())).status == "pass"


def test_ratio_is_the_threshold():
    groups = {str(g): {"all_gather": 22.0, "reduce_scatter": 22.5} for g in range(4)}
    groups["0"] = {"all_gather": 15.0, "reduce_scatter": 15.0}  # 0.68 x median
    r = _result(groups=groups)
    assert judge(_log(r), ratio=0.6).status == "pass"
    assert judge(_log(r), ratio=0.75).status == "exclude"


@pytest.mark.parametrize(
    "log, status, exclude",
    [
        # 2026-10-28, 125 nodes: rail 2 at 11.9 GB/s; as01r5b17 receives one slow pair and sends
        # the other (as01r5b17 was the bad NIC; as01r5b19 is only its ring neighbour)
        ("mn5_20261028_rail2_as01r5b17.txt", "exclude", ["as01r5b17"]),
        # the same allocation without it: all groups at 22.3-22.6 GB/s
        ("mn5_20261028_clean.txt", "pass", []),
    ],
)
def test_real_marenostrum_logs(log, status, exclude):
    """The earlier one-off benchmark's text format (scripts/korbi/pool/fsdp_comm_bench.py)."""
    v = judge((DATA / log).read_text())
    assert (v.status, v.exclude) == (status, exclude), v


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_bench_on_cpu_two_agents(tmp_path):
    """The real benchmark, 2 'nodes' x 2 ranks on gloo: one result on rank 0, judged pass,
    node names from SLURMD_NODENAME, every ring pair present."""
    port = str(_free_port())
    procs = []
    for node in range(2):
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes",
            "2",
            "--nproc-per-node",
            "2",
            "--node-rank",
            str(node),
            "--master-addr",
            "127.0.0.1",
            "--master-port",
            port,
            "-m",
            "oellm_autoexp.node_health.bench",
            "--device",
            "cpu",
            "--coll-mb",
            "4",
            "--ring-mb",
            "4",
            "--matmul-n",
            "128",
            "--iters",
            "2",
        ]
        env = {**os.environ, "SLURMD_NODENAME": f"cpu{node}", "PYTHONPATH": str(REPO)}
        out = (tmp_path / f"node{node}.log").open("w")
        procs.append(
            subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, cwd=tmp_path)
        )
    assert [p.wait(timeout=180) for p in procs] == [0, 0], (tmp_path / "node0.log").read_text()[
        -3000:
    ]
    text = (tmp_path / "node0.log").read_text()
    assert PREFIX not in (tmp_path / "node1.log").read_text(), "only rank 0 reports"
    line = next(x for x in text.splitlines() if x.startswith(PREFIX))
    r = json.loads(line[len(PREFIX) :])
    assert r["nodes"] == ["cpu0", "cpu1"] and r["gpus_per_node"] == 2
    assert sorted(r["compute"]) == ["cpu0", "cpu1"] and all(
        len(v) == 2 for v in r["compute"].values()
    )
    assert sorted(r["groups"]) == ["0", "1"]
    assert sorted((a, b, g) for a, b, g, _ in r["pairs"]) == [
        ("cpu0", "cpu1", 0),
        ("cpu0", "cpu1", 1),
        ("cpu1", "cpu0", 0),
        ("cpu1", "cpu0", 1),
    ]
    assert all(bw > 0 for *_, bw in r["pairs"])
    # CPU timings are noisy: only that the verdict reads it
    assert judge(text, ratio=0.0).status == "pass"
