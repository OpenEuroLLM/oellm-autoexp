import pytest
from oellm_autoexp.hydra_staged_sweep.dag_resolver import (
    resolve_sweep_with_dag,
    extract_sibling_patterns,
    build_dependency_dag_from_points,
    config_to_cmdline,
)
from oellm_autoexp.hydra_staged_sweep.config.schema import StagedSweepRoot, ConfigSetup
from oellm_autoexp.hydra_staged_sweep.expander import SweepPoint


def test_extract_sibling_patterns_nested():
    params = {"dict": {"a": "${sibling.s1.x}"}, "list": ["${sibling.s2.y}", "plain"]}
    assert extract_sibling_patterns(params) == {"s1", "s2"}


def test_resolve_circular_dependency():
    p0 = SweepPoint(
        index=0,
        parameters={"stage": "A", "ref": "${sibling.B.x}"},
        group_path=(0, 0),
        stage_path=(False, False),
    )
    p1 = SweepPoint(
        index=1,
        parameters={"stage": "B", "ref": "${sibling.A.x}"},
        group_path=(0, 1),
        stage_path=(False, True),
    )

    config = StagedSweepRoot()
    setup = ConfigSetup(pwd=".", config_path=".", config_dir=".")

    with pytest.raises(ValueError, match="Circular dependencies detected"):
        resolve_sweep_with_dag(config, [p0, p1], setup)


def test_build_dag_warning(caplog):
    # Case where sibling pattern exists but no sibling found
    p = SweepPoint(
        index=0, parameters={"ref": "${sibling.missing.x}"}, group_path=(0,), stage_path=(True,)
    )
    build_dependency_dag_from_points({0: p})
    assert "No sibling found for requested stage_pattern: missing" in caplog.text


def test_sibling_match_ignores_stage_path():
    p0 = SweepPoint(
        index=0, parameters={"stage": "A"}, group_path=(0, 0), stage_path=(False, False)
    )
    p1 = SweepPoint(
        index=1,
        parameters={"stage": "B", "ref": "${sibling.A.x}"},
        group_path=(0, 1),
        stage_path=(False, True),
    )
    p2 = SweepPoint(
        index=2, parameters={"stage": "A"}, group_path=(1, 0), stage_path=(False, False)
    )
    dag = build_dependency_dag_from_points({0: p0, 1: p1, 2: p2})
    assert (0, 1) in dag.edges()
    assert (2, 1) not in dag.edges()


def test_config_to_cmdline_edge_cases():
    # List conversion
    res = config_to_cmdline(["a", "b"], prefix="l")
    assert "l=[0,1]" in res or ("l.0=a" in res and "l.1=b" in res)

    # None conversion
    assert "n=null" in config_to_cmdline(None, prefix="n")

    # Unescape interpolations in strings
    assert 'val="${interp}"' in config_to_cmdline(r"${interp}", prefix="val")

    # Prefix handling
    assert 'foo.bar="val"' in config_to_cmdline({"bar": "val"}, prefix="foo")
