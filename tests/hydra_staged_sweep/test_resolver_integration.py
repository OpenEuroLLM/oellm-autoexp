import pytest
import yaml
from dataclasses import dataclass, field
from compoconf import ConfigInterface, asdict

from oellm_autoexp.hydra_staged_sweep.dag_resolver import resolve_sweep_with_dag
from oellm_autoexp.hydra_staged_sweep.config.schema import StagedSweepRoot, ConfigSetup, SweepConfig
from oellm_autoexp.hydra_staged_sweep.config.resolvers import register_default_resolvers
from oellm_autoexp.hydra_staged_sweep.config.loader import load_config_reference
from oellm_autoexp.hydra_staged_sweep.expander import SweepPoint, expand_sweep


@dataclass(kw_only=True)
class MyProjectConfig(ConfigInterface):
    name: str = "test-project"


@dataclass(kw_only=True)
class MyRootConfig(StagedSweepRoot):
    project: MyProjectConfig = field(default_factory=MyProjectConfig)
    some_param: str = "default"


def test_resolve_simple_sweep(tmp_path):
    # Setup
    config = MyRootConfig()
    config.sweep = SweepConfig()

    points = [
        SweepPoint(
            index=0, parameters={"some_param": "val1"}, group_path=(0,), stage_path=(False,)
        ),
        SweepPoint(
            index=1, parameters={"some_param": "val2"}, group_path=(1,), stage_path=(False,)
        ),
    ]

    config_path = tmp_path / "test.yaml"
    with open(str(config_path), "w") as fp:
        yaml.dump(asdict(config), fp)

    setup = ConfigSetup(pwd="/tmp", config_path=config_path)

    jobs = resolve_sweep_with_dag(config, points, setup, config_class=MyRootConfig)

    assert len(jobs) == 2
    # Verify job 0
    assert jobs[0].config.some_param == "val1"
    assert jobs[0].config.index == 0

    # Verify job 1
    assert jobs[1].config.some_param == "val2"
    assert jobs[1].config.index == 1


def test_resolve_sweep_filter_skips_points(tmp_path):
    config = MyRootConfig()
    config.sweep = SweepConfig(filter='${oc.eval:\'"${some_param}" == "val2"\'}')
    register_default_resolvers(force=True)

    points = [
        SweepPoint(
            index=0, parameters={"some_param": "val1"}, group_path=(0,), stage_path=(False,)
        ),
        SweepPoint(
            index=1, parameters={"some_param": "val2"}, group_path=(1,), stage_path=(False,)
        ),
    ]

    config_path = tmp_path / "test.yaml"
    with open(str(config_path), "w") as fp:
        yaml.dump(asdict(config), fp)

    setup = ConfigSetup(pwd="/tmp", config_path=config_path)

    jobs = resolve_sweep_with_dag(config, points, setup, config_class=MyRootConfig)

    assert len(jobs) == 1
    assert jobs[0].config.index == 1


def test_resolve_sweep_filter_non_bool_raises(tmp_path):
    config = MyRootConfig()
    config.sweep = SweepConfig(filter="not-bool")

    points = [
        SweepPoint(index=0, parameters={"some_param": "val1"}, group_path=(0,), stage_path=(False,))
    ]

    config_path = tmp_path / "test.yaml"
    with open(str(config_path), "w") as fp:
        yaml.dump(asdict(config), fp)

    setup = ConfigSetup(pwd="/tmp", config_path=config_path)

    with pytest.raises(ValueError, match="sweep.filter must resolve to a bool"):
        resolve_sweep_with_dag(config, points, setup, config_class=MyRootConfig)


def test_resolve_sweep_filter_resolution_error(tmp_path):
    config = MyRootConfig()
    config.sweep = SweepConfig(filter="\\${missing:1}")

    points = [
        SweepPoint(index=0, parameters={"some_param": "val1"}, group_path=(0,), stage_path=(False,))
    ]
    config_path = tmp_path / "test.yaml"
    with open(str(config_path), "w") as fp:
        yaml.dump(asdict(config), fp)

    setup = ConfigSetup(pwd="/tmp", config_path=config_path)

    with pytest.raises(ValueError, match="sweep.filter must resolve to a bool."):
        resolve_sweep_with_dag(config, points, setup, config_class=MyRootConfig)


def test_resolve_sweep_group_path_mismatch_raises():
    config = MyRootConfig()
    config.sweep = SweepConfig(
        type="product",
        groups=[{"type": "product", "params": {"some_param": ["val1"]}}],
    )
    points = [
        SweepPoint(
            index=0, parameters={"some_param": "val1"}, group_path=(1, 0), stage_path=(False,)
        )
    ]
    setup = ConfigSetup(pwd="/tmp", config_name="conf", config_dir="/tmp")

    with pytest.raises(ValueError, match="Unable to match group_path"):
        resolve_sweep_with_dag(config, points, setup, config_class=MyRootConfig)


def test_resolve_sweep_filter_none_entry(tmp_path):
    config = MyRootConfig()
    config.sweep = SweepConfig(
        filter=True,
        type="product",
        groups=[{"type": "product", "params": {"some_param": ["val1"]}, "filter": None}],
    )
    register_default_resolvers(force=True)

    config_path = tmp_path / "test.yaml"
    with open(str(config_path), "w") as fp:
        yaml.dump(asdict(config), fp)

    setup = ConfigSetup(pwd="/tmp", config_path=config_path)
    points = expand_sweep(config.sweep)

    jobs = resolve_sweep_with_dag(config, points, setup, config_class=MyRootConfig)
    assert len(jobs) == 1


def test_resolve_sweep_filter_by_sibling(tmp_path):
    config = MyRootConfig()
    config.sweep = SweepConfig(
        type="product",
        groups=[
            {
                "type": "list",
                "configs": [
                    {"stage": "stage1", "some_param": "val1"},
                    {"stage": "stage2", "some_param": "\\${sibling.stage1.some_param}2"},
                ],
            }
        ],
        filter="${oc.eval:'\\'${some_param}\\'!=\\'val12\\''}",
    )
    register_default_resolvers(force=True)

    config_path = tmp_path / "test.yaml"
    with open(str(config_path), "w") as fp:
        yaml.dump(asdict(config), fp)

    setup = ConfigSetup(pwd="/tmp", config_path=config_path)

    points = expand_sweep(config.sweep)
    jobs = resolve_sweep_with_dag(config, points, setup, config_class=MyRootConfig)
    assert len(jobs) == 1


def test_resolve_sweep_filter_unresolved_then_false(tmp_path):
    points = [
        SweepPoint(index=0, parameters={"some_param": "val1"}, group_path=(0,), stage_path=(False,))
    ]
    config = MyRootConfig()
    config.sweep = SweepConfig(
        filter=False,
        type="list",
        groups=[{"type": "product", "params": {"some_param": ["val1"]}}],
    )
    config_path = tmp_path / "test.yaml"
    with open(str(config_path), "w") as fp:
        yaml.dump(asdict(config), fp)
    config = load_config_reference(config_path=config_path, config_class=MyRootConfig)
    config.sweep = SweepConfig(filter="expr")

    setup = ConfigSetup(pwd="/tmp", config_path=config_path)
    with pytest.raises(ValueError, match="sweep.filter must resolve to a bool."):
        resolve_sweep_with_dag(config, points, setup, config_class=MyRootConfig)

    config = load_config_reference(config_path=config_path, config_class=MyRootConfig)
    config.sweep = SweepConfig(filter="${expr}")

    setup = ConfigSetup(pwd="/tmp", config_path=config_path)
    with pytest.raises(ValueError, match="sweep.filter must resolve to a bool."):
        resolve_sweep_with_dag(config, points, setup, config_class=MyRootConfig)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_path:
        test_resolve_sweep_filter_by_sibling(Path(tmp_path))
