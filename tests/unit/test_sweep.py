from hydra_staged_sweep.config.schema import SweepConfig
from oellm_autoexp.sweep.expander import expand_sweep


def test_expand_sweep_simple():
    cfg = SweepConfig(type="list", groups=[{"type": "product", "params": {"a": [1, 2]}}])
    points = expand_sweep(cfg)
    assert len(points) == 2
