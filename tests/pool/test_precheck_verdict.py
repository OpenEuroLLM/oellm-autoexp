"""scripts/pool/precheck_verdict.py: which nodes a pool host drops after its pre-run check."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "precheck_verdict", Path(__file__).resolve().parents[2] / "scripts/pool/precheck_verdict.py"
)
pv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pv)

NODES = [f"n{i}" for i in range(6)]


def ok(node, tflops=None):
    return f"[nodecheck] OK {node} mem=500GiB" + (f" gemm_tflops={tflops}" if tflops else "")


def excluded(lines, need, nodes=NODES, tol=0.10):
    return [n for n, _ in pv.verdict(lines, nodes, need, tol)]


def test_fail_and_silent_nodes_are_dropped():
    lines = [
        "[nodecheck] FAIL n0: row remapping pending",
        *(ok(n, 900) for n in NODES[1:5]),  # n5 never reports
    ]
    assert excluded(lines, need=4) == ["n0", "n5"]


def test_a_check_failing_more_nodes_than_the_spares_is_not_trusted():
    # a collective probe that breaks takes every node with it: never let it stop the run
    warnings = []
    lines = [f"[nodecheck] FAIL {n}: ib probe: NCCL timeout" for n in NODES]
    assert pv.verdict(lines, NODES, 4, 0.1, warnings.append) == []
    assert warnings and "not trusted" in warnings[0]
    lines = ["[nodecheck] FAIL n0: x", "[nodecheck] FAIL n1: x", *(ok(n, 900) for n in NODES[2:])]
    assert excluded(lines, need=5) == []  # 2 failed, 1 spare


def ib(node, gbps):
    return f"[nodecheck] OK {node} ib_gbps={gbps} ib_rounds={gbps},{gbps}"


def test_a_slow_link_is_dropped_like_a_slow_gpu():
    lines = [*(ok(n, 900) for n in NODES), *(ib(n, 45) for n in NODES[1:]), ib("n0", 12)]
    assert excluded(lines, need=4) == ["n0"]


def test_the_worst_shortfall_over_both_metrics_goes_first():
    lines = [ok("n0", 700), ok("n1", 890), *(ok(n, 900) for n in NODES[2:]),
             ib("n1", 10), *(ib(n, 45) for n in NODES if n != "n1")]  # fmt: skip
    assert excluded(lines, need=5) == ["n1"]  # 1 spare: n1 (78% short) before n0 (22%)


def test_slowest_dropped_only_as_far_as_the_spares_allow():
    lines = [
        ok("n0", 400),
        ok("n1", 500),
        ok("n2", 880),
        ok("n3", 900),
        ok("n4", 910),
        ok("n5", 905),
    ]
    assert excluded(lines, need=4) == ["n0", "n1"]  # 2 spares, both slow ones
    assert excluded(lines, need=5) == ["n0"]  # 1 spare: the slowest
    assert excluded(lines, need=6) == []  # no spare: the run keeps every node


def test_within_tolerance_is_not_slow():
    lines = [ok(n, 900 - 10 * i) for i, n in enumerate(NODES)]  # 900 .. 850, median 875
    assert excluded(lines, need=1) == []


def test_a_gemm_fail_line_drops_the_node_and_its_speed_is_ignored():
    lines = [ok(n, 900) for n in NODES] + ["[nodecheck] FAIL n2: gemm probe: CUDA error"]
    assert excluded(lines, need=1) == ["n2"]


def test_no_speed_verdict_from_fewer_than_three_healthy_nodes():
    nodes = ["a", "b", "c"]
    lines = ["[nodecheck] FAIL a: x", ok("b", 100), ok("c", 900)]
    assert excluded(lines, need=1, nodes=nodes) == ["a"]


def test_lines_of_other_nodes_and_noise_are_ignored():
    lines = ["srun: error: something", ok("zz", 1), *(ok(n, 900) for n in NODES)]
    assert excluded(lines, need=1) == []
