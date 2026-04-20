from pathlib import Path
from dataclasses import dataclass
from hydra_staged_sweep.config.loader import (
    load_config_reference,
)
from compoconf import NonStrictDataclass


@dataclass(kw_only=True, init=False)
class VarClass(NonStrictDataclass):
    pass


def test_load_config1():
    cfg = load_config_reference(
        config_path=Path(__file__).parent / "configs" / "test1.yaml", config_class=VarClass
    )
    assert cfg.a == 1
    assert cfg.b is None
    assert cfg.c == 1
    assert cfg.d == "123 {{script}}"
    assert cfg.d.replace("{{script}}", "4") == "123 4"
    assert cfg.e == "{{script}}"
