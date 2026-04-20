import networkx as nx
from hydra_staged_sweep.dag_resolver import (
    extract_sibling_patterns,
    find_sibling_by_group_path,
    build_dependency_dag_from_points,
)
from hydra_staged_sweep.expander import SweepPoint


def test_extract_sibling_patterns():
    params = {
        "lr": "${sibling.stable.lr}",
        "other": "constant",
        "nested": {"path": "${sibling.decay.path}"},
    }
    patterns = extract_sibling_patterns(params)
    assert patterns == {"stable", "decay"}


def test_find_sibling_by_group_path():
    # Point A: stage=stable, group=(0, 0), stage_path=(False, False)
    # Point B: stage=decay,  group=(0, 1), stage_path=(False, True) -> depends on stable

    p_stable = SweepPoint(
        index=0,
        parameters={"stage": "stable", "lr": 0.01},
        group_path=(0, 0),
        stage_path=(False, False),
    )
    p_decay = SweepPoint(
        index=1,
        parameters={"stage": "decay", "lr": "${sibling.stable.lr}"},
        group_path=(0, 1),
        stage_path=(False, True),
    )

    all_points = {0: p_stable, 1: p_decay}

    # Looking for 'stable' sibling for p_decay
    found = find_sibling_by_group_path(p_decay, all_points, "stable")
    assert found == p_stable


def test_build_dependency_dag():
    p0 = SweepPoint(
        index=0, parameters={"stage": "A"}, group_path=(0, 0), stage_path=(False, False)
    )
    p1 = SweepPoint(
        index=1,
        parameters={"stage": "B", "ref": "${sibling.A.x}"},
        group_path=(0, 1),
        stage_path=(False, True),
    )

    points = {0: p0, 1: p1}
    dag = build_dependency_dag_from_points(points)

    assert dag.has_edge(0, 1)
    assert nx.is_directed_acyclic_graph(dag)
