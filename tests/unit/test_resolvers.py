from omegaconf import OmegaConf

from oellm_autoexp.config.resolvers import register_default_resolvers


def test_default_resolvers_evaluate_expressions():
    register_default_resolvers(force=True)

    cfg = OmegaConf.create(
        {
            "product": "${oc.mul:2,3,4}",
            "ceil": "${oc.ceil_div:5,2}",
            "bool_int": "${oc.int:True}",
        }
    )
    resolved = OmegaConf.to_object(cfg)
    assert resolved["product"] == 24
    assert resolved["ceil"] == 3
    assert resolved["bool_int"] == 1


def test_register_default_resolvers_idempotent():
    register_default_resolvers()
    register_default_resolvers()
    cfg = OmegaConf.create({"val": "${oc.addi:1,2}"})
    assert OmegaConf.to_object(cfg)["val"] == 3
