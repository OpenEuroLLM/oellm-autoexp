from omegaconf import OmegaConf


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
