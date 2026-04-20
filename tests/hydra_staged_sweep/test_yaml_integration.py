import textwrap
from dataclasses import dataclass, field
from compoconf import ConfigInterface
from hydra_staged_sweep.config.schema import StagedSweepRoot, ConfigSetup
from hydra_staged_sweep.expander import expand_sweep
from hydra_staged_sweep.dag_resolver import resolve_sweep_with_dag
from hydra_staged_sweep.config.loader import load_config


@dataclass(kw_only=True)
class MyProjectConfig(ConfigInterface):
    name: str = ""
    base_output_dir: str = ""
    log_path: str = ""
    log_path_current: str = ""


@dataclass(kw_only=True)
class MyRootConfig(StagedSweepRoot):
    project: MyProjectConfig = field(default_factory=MyProjectConfig)
    learning_rate: float = 0.0
    batch_size: int = 0
    train_iters: int = 0
    load_path: str | None = None


def test_yaml_integration(tmp_path):
    # Use raw string to avoid Python unescaping
    # We want YAML to contain: load_path: "\\${sibling...}"
    config_yaml = textwrap.dedent(r"""
        project:
          name: "demo-experiment"
          base_output_dir: "${project.name}"
          log_path: "${project.base_output_dir}/%j.out"
          log_path_current: "${project.base_output_dir}/latest.out"

        sweep:
          type: "product"
          groups:
            - type: "product"
              params:
                learning_rate: [1.0e-4, 5.0e-4]
                batch_size: [32, 64]

            - type: "list"
              configs:
                - stage: "stable"
                  train_iters: 1000

                - stage: "decay"
                  train_iters: 200
                  load_path: "\\${sibling.stable.project.base_output_dir}/checkpoints"
    """)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)

    # Load
    root_config = load_config(config_path, config_class=MyRootConfig)

    # Expand
    points = expand_sweep(root_config.sweep)
    assert len(points) == 8  # 2 LRs * 2 BS * 2 Stages

    # Resolve
    setup = ConfigSetup(
        pwd=str(tmp_path),
        config_path=str(config_path),
        config_dir=str(tmp_path),
    )

    plans = resolve_sweep_with_dag(root_config, points, setup, config_class=MyRootConfig)

    assert len(plans) == 8

    # Check for resolution of sibling path
    decay_job = next(
        p
        for p in plans
        if p.config.stage == "decay"
        and p.config.learning_rate == 1.0e-4
        and p.config.batch_size == 32
    )
    stable_job = next(
        p
        for p in plans
        if p.config.stage == "stable"
        and p.config.learning_rate == 1.0e-4
        and p.config.batch_size == 32
    )

    expected_path = f"{stable_job.config.project.base_output_dir}/checkpoints"
    assert decay_job.config.load_path == expected_path
    assert "demo-experiment/checkpoints" in decay_job.config.load_path


def test_yaml_integration_filter(tmp_path):
    config_yaml = textwrap.dedent(r"""
        project:
          name: "demo-experiment"
          base_output_dir: "${project.name}"
          log_path: "${project.base_output_dir}/%j.out"
          log_path_current: "${project.base_output_dir}/latest.out"

        sweep:
          type: "product"
          groups:
            - type: "product"
              params:
                learning_rate: [1.0e-4, 5.0e-4]
                batch_size: [32, 64]
          filter: "\\${oc.eval:'\\${learning_rate} < 0.001 and \\${batch_size} == 32'}"
    """)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)

    root_config = load_config(config_path, config_class=MyRootConfig)
    points = expand_sweep(root_config.sweep)
    assert len(points) == 4

    setup = ConfigSetup(
        pwd=str(tmp_path),
        config_path=str(config_path),
        config_dir=str(tmp_path),
    )

    plans = resolve_sweep_with_dag(root_config, points, setup, config_class=MyRootConfig)
    assert len(plans) == 2
    assert all(p.config.batch_size == 32 for p in plans)
    assert sorted(p.config.learning_rate for p in plans) == [1.0e-4, 5.0e-4]


def test_yaml_integration_filter_with_sibling(tmp_path):
    config_yaml = textwrap.dedent(r"""
        project:
          name: "demo-experiment"
          base_output_dir: "${project.name}"
          log_path: "${project.base_output_dir}/%j.out"
          log_path_current: "${project.base_output_dir}/latest.out"

        sweep:
          type: "product"
          groups:
            - type: "product"
              params:
                learning_rate: [1.0e-4, 5.0e-4]
                batch_size: [32, 64]

            - type: "list"
              configs:
                - stage: "stable"
                  train_iters: 1000

                - stage: "decay"
                  train_iters: 200
                  load_path: "\\${sibling.stable.project.base_output_dir}/checkpoints"
          filter: "\\${oc.eval:'\\${learning_rate} < 0.001 and \\${batch_size} == 32'}"
    """)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)

    root_config = load_config(config_path, config_class=MyRootConfig)
    points = expand_sweep(root_config.sweep)
    assert len(points) == 8

    setup = ConfigSetup(
        pwd=str(tmp_path),
        config_path=str(config_path),
        config_dir=str(tmp_path),
    )

    plans = resolve_sweep_with_dag(root_config, points, setup, config_class=MyRootConfig)
    assert len(plans) == 4
    assert all(p.config.batch_size == 32 for p in plans)

    decay_job = next(
        p
        for p in plans
        if p.config.stage == "decay"
        and p.config.learning_rate == 1.0e-4
        and p.config.batch_size == 32
    )
    stable_job = next(
        p
        for p in plans
        if p.config.stage == "stable"
        and p.config.learning_rate == 1.0e-4
        and p.config.batch_size == 32
    )
    expected_path = f"{stable_job.config.project.base_output_dir}/checkpoints"
    assert decay_job.config.load_path == expected_path


def test_yaml_integration_group_filter_with_sibling(tmp_path):
    config_yaml = textwrap.dedent(r"""
        project:
          name: "demo-experiment"
          base_output_dir: "${project.name}"
          log_path: "${project.base_output_dir}/%j.out"
          log_path_current: "${project.base_output_dir}/latest.out"

        sweep:
          type: "product"
          groups:
            - type: "product"
              params:
                learning_rate: [1.0e-4, 5.0e-4]
                batch_size: [32, 64]
              filter: "\\${oc.eval:'\\${batch_size} == 32'}"

            - type: "list"
              configs:
                - stage: "stable"
                  train_iters: 1000

                - stage: "decay"
                  train_iters: 200
                  load_path: "\\${sibling.stable.project.base_output_dir}/checkpoints"
    """)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)

    root_config = load_config(config_path, config_class=MyRootConfig)
    points = expand_sweep(root_config.sweep)
    assert len(points) == 8

    setup = ConfigSetup(
        pwd=str(tmp_path),
        config_path=str(config_path),
        config_dir=str(tmp_path),
    )

    plans = resolve_sweep_with_dag(root_config, points, setup, config_class=MyRootConfig)
    assert len(plans) == 4
    assert all(p.config.batch_size == 32 for p in plans)

    decay_job = next(
        p
        for p in plans
        if p.config.stage == "decay"
        and p.config.learning_rate == 1.0e-4
        and p.config.batch_size == 32
    )
    stable_job = next(
        p
        for p in plans
        if p.config.stage == "stable"
        and p.config.learning_rate == 1.0e-4
        and p.config.batch_size == 32
    )
    expected_path = f"{stable_job.config.project.base_output_dir}/checkpoints"
    assert decay_job.config.load_path == expected_path


def test_yaml_integration_filter_v2(tmp_path):
    config_yaml = textwrap.dedent(r"""
        project:
          name: "demo-experiment"
          base_output_dir: "${project.name}"
          log_path: "${project.base_output_dir}/%j.out"
          log_path_current: "${project.base_output_dir}/latest.out"

        sweep:
          type: "product"
          groups:
            - type: "product"
              params:
                learning_rate: [1.0e-4, 5.0e-4]
                batch_size: [32, 64]
          filter: "\\${oc.eval:'\\${learning_rate} < 0.001 and \\${batch_size} == 32'}"
    """)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)

    root_config = load_config(config_path, config_class=MyRootConfig)
    points = expand_sweep(root_config.sweep)
    assert len(points) == 4

    setup = ConfigSetup(
        pwd=str(tmp_path),
        config_path=str(config_path),
        config_dir=str(tmp_path),
    )

    plans = resolve_sweep_with_dag(root_config, points, setup, config_class=MyRootConfig)
    assert len(plans) == 2
    assert all(p.config.batch_size == 32 for p in plans)
    assert sorted(p.config.learning_rate for p in plans) == [1.0e-4, 5.0e-4]


def test_yaml_integration_filter_extended_with_sibling(tmp_path):
    config_yaml = textwrap.dedent(r"""
        project:
          name: "demo-experiment"
          base_output_dir: "${project.name}"
          log_path: "${project.base_output_dir}/%j.out"
          log_path_current: "${project.base_output_dir}/latest.out"

        sweep:
          type: "product"
          groups:
            - type: "product"
              params:
                learning_rate: [1.0e-4, 2.0e-4]
                batch_size: [32, 64]

            - type: "list"
              configs:
                - stage: "stable"
                  train_iters: 1000

                - stage: "decay"
                  train_iters: 200
                  load_path: "\\${sibling.stable.project.base_output_dir}/checkpoints"
          filter: "\\${oc.eval:'\\${learning_rate}*\\${batch_size} < 0.01'}"
    """)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)

    root_config = load_config(config_path, config_class=MyRootConfig)
    points = expand_sweep(root_config.sweep)
    assert len(points) == 8

    setup = ConfigSetup(
        pwd=str(tmp_path),
        config_path=str(config_path),
        config_dir=str(tmp_path),
    )

    plans = resolve_sweep_with_dag(root_config, points, setup, config_class=MyRootConfig)
    assert len(plans) == 6

    bad_job = list(
        p for p in plans if p.config.learning_rate == 2.0e-4 and p.config.batch_size == 64
    )
    assert len(bad_job) == 0
