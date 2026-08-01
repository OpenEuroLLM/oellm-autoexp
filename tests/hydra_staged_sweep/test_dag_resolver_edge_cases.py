import pytest
from oellm_autoexp.hydra_staged_sweep.dag_resolver import (
    resolve_sweep_with_dag,
    extract_sibling_patterns,
    build_dependency_dag_from_points,
    config_to_cmdline,
)
from oellm_autoexp.hydra_staged_sweep.config.schema import StagedSweepRoot, ConfigSetup, SweepConfig
from oellm_autoexp.hydra_staged_sweep.expander import SweepPoint, expand_sweep


def _ladder(gbs: int, budget: int) -> dict:
    """A stable + one WSD cooldown, sharing a per-ladder gbs default."""
    return {
        "type": "list",
        "defaults": {"gbs": gbs},
        "configs": [
            {"stage": "stable"},
            {
                "type": "list",
                "defaults": {"stage": "decay", "load": "${sibling.stable.dir}"},
                "configs": [{"tokens": budget}],
            },
        ],
    }


def test_multiladder_cooldown_branches_from_own_stable():
    """Regression for the nested-group stage-detection bug: two ladders that
    differ only in gbs live in one sweep file. Each cooldown must branch from
    its OWN ladder's stable (same gbs), not the other ladder's. This holds only
    if the ladder-selector group-path position is NOT flagged as a stage axis
    (it merely contains stable/cooldown sub-sweeps)."""
    config = SweepConfig(
        type="product",
        groups=[
            {"type": "list", "configs": [{"variant": "A"}]},  # architecture axis
            {"type": "list", "configs": [_ladder(128, 20), _ladder(256, 80)]},  # two ladders
        ],
    )
    points = expand_sweep(config)

    def find(stage: str, gbs: int) -> SweepPoint:
        return next(
            p for p in points
            if p.parameters.get("stage") == stage and p.parameters.get("gbs") == gbs
        )

    stable128, stable256 = find("stable", 128), find("stable", 256)
    decay128, decay256 = find("decay", 128), find("decay", 256)

    # Distinct ladders must NOT collapse to the same sibling key.
    dag = build_dependency_dag_from_points({p.index: p for p in points})
    assert (stable128.index, decay128.index) in dag.edges()
    assert (stable256.index, decay256.index) in dag.edges()
    # ...and a cooldown must NOT branch from the other ladder's stable.
    assert (stable256.index, decay128.index) not in dag.edges()
    assert (stable128.index, decay256.index) not in dag.edges()


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
