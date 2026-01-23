from omegaconf import OmegaConf

from oellm_autoexp.config.resolvers import register_default_resolvers


def test_resolver_join():
    cfg = {"a": ["1", "2", "3"], "c": "${oc.join:'.',${a}}"}

    ocfg = OmegaConf.create(cfg)
    cfg = OmegaConf.to_container(ocfg, resolve=True)
    assert cfg["c"] == "1.2.3"


def test_template():
    cfg = {"a": "a", "b": "${oc.tmpl:'$%=\\'$%\\'',${a}}"}
    ocfg = OmegaConf.create(cfg)
    cfg = OmegaConf.to_container(ocfg, resolve=True)
    assert cfg["b"] == "$a='$a'"


def test_template_map():
    cfg = {"a": ["a", "b", "c"], "b": "${oc.maptmpl:'$%=\\'$%\\'',${a}}"}
    ocfg = OmegaConf.create(cfg)
    cfg = OmegaConf.to_container(ocfg, resolve=True)
    assert cfg["b"] == ["$a='$a'", "$b='$b'", "$c='$c'"]


def test_split():
    cfg = {"a": "123 1312 312 13 1241 1214", "b": "${oc.split:${a},' '}"}
    ocfg = OmegaConf.create(cfg)
    cfg = OmegaConf.to_container(ocfg, resolve=True)
    assert cfg["b"] == ["123", "1312", "312", "13", "1241", "1214"]


def test_default_resolvers_evaluate_expressions():
    register_default_resolvers(force=True)

    cfg = OmegaConf.create(
        {
            "product": "${oc.mul:2,3,4}",
            "ceil": "${oc.cdivi:5,2}",
            "difference": "${oc.sub:10,4}",
            "ratio": "${oc.div:9,3}",
            "bool_int": "${oc.int:True}",
        }
    )
    resolved = OmegaConf.to_object(cfg)
    assert resolved["product"] == 24
    assert resolved["ceil"] == 3
    assert resolved["difference"] == 6.0
    assert resolved["ratio"] == 3.0
    assert resolved["bool_int"] == 1


def test_register_default_resolvers_idempotent():
    register_default_resolvers()
    register_default_resolvers()
    cfg = OmegaConf.create({"val": "${oc.addi:1,2}"})
    assert OmegaConf.to_object(cfg)["val"] == 3
