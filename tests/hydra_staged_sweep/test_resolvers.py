import pytest
from omegaconf import OmegaConf
from hydra_staged_sweep.config.resolvers import register_default_resolvers


@pytest.fixture(autouse=True)
def setup_resolvers():
    register_default_resolvers(force=True)


def test_mul():
    c = OmegaConf.create({"val": "${oc.mul:2,3,4}"})
    assert c.val == 24.0
    c2 = OmegaConf.create({"val": "${oc.mul:2,invalid,4}"})
    assert c2.val == 8.0


def test_muli():
    c = OmegaConf.create({"val": "${oc.muli:2,3,4}"})
    assert c.val == 24
    c2 = OmegaConf.create({"val": "${oc.muli:2,invalid,4}"})
    assert c2.val == 8


def test_add():
    c = OmegaConf.create({"val": "${oc.add:2,3,4}"})
    assert c.val == 9.0
    c2 = OmegaConf.create({"val": "${oc.add:2,invalid,4}"})
    assert c2.val == 6.0


def test_addi():
    c = OmegaConf.create({"val": "${oc.addi:2,3,4}"})
    assert c.val == 9
    c2 = OmegaConf.create({"val": "${oc.addi:2,invalid,4}"})
    assert c2.val == 6


def test_sub():
    c = OmegaConf.create({"val": "${oc.sub:10,4}"})
    assert c.val == 6.0
    c2 = OmegaConf.create({"val": "${oc.sub:10,invalid}"})
    assert c2.val == 0.0


def test_subi():
    c = OmegaConf.create({"val": "${oc.subi:10,4}"})
    assert c.val == 6
    c2 = OmegaConf.create({"val": "${oc.subi:10,invalid}"})
    assert c2.val == 0


def test_div():
    c = OmegaConf.create({"val": "${oc.div:10,2}"})
    assert c.val == 5.0
    c2 = OmegaConf.create({"val": "${oc.div:10,0}"})
    assert c.val == 5.0  # Wait, c2.val
    assert c2.val == 0.0

    # List division
    c3 = OmegaConf.create({"l": [10, 20], "r": [2, 5], "val": "${oc.div:${l},${r}}"})
    assert list(c3.val) == [5.0, 4.0]

    c4 = OmegaConf.create({"l": [10, 20], "r": 2, "val": "${oc.div:${l},${r}}"})
    assert list(c4.val) == [5.0, 10.0]


def test_divi():
    c = OmegaConf.create({"val": "${oc.divi:10,3}"})
    assert c.val == 3
    c2 = OmegaConf.create({"val": "${oc.divi:10,0}"})
    assert c2.val == 0

    # List division
    c3 = OmegaConf.create({"l": [10, 20], "r": [2, 5], "val": "${oc.divi:${l},${r}}"})
    assert list(c3.val) == [5, 4]

    c4 = OmegaConf.create({"l": [10, 20], "r": 2, "val": "${oc.divi:${l},${r}}"})
    assert list(c4.val) == [5, 10]


def test_cdivi():
    c = OmegaConf.create({"val": "${oc.cdivi:10,3}"})
    assert c.val == 4
    c2 = OmegaConf.create({"val": "${oc.cdivi:10,0}"})
    assert c2.val == 0
    c3 = OmegaConf.create({"val": "${oc.cdivi:invalid,3}"})
    assert c3.val == 0


def test_sqrt():
    c = OmegaConf.create({"val": "${oc.sqrt:16}"})
    assert c.val == 4.0


def test_slice():
    c = OmegaConf.create({"val": "${oc.slice:hello,0,3}"})
    assert c.val == "hel"


def test_mul_round_int():
    c = OmegaConf.create({"val": "${oc.mul_round_int:10,1.5,8}"})
    assert c.val == 16


def test_concat():
    c = OmegaConf.create({"val": "${oc.concat:foo,bar}"})
    assert c.val == "foobar"


def test_int_cast():
    c = OmegaConf.create({"val": "${oc.int:123}"})
    assert c.val == 123
    c2 = OmegaConf.create({"val": "${oc.int:True}"})
    assert c2.val == 1
    c3 = OmegaConf.create({"val": "${oc.int:invalid}"})
    assert c3.val == 1  # bool("invalid") is True, int(True) is 1


def test_dict_merge():
    c = OmegaConf.create(
        {"d1": {"a": 1}, "d2": {"b": 2}, "val": "${oc.dict_merge:${d1},${d2},null}"}
    )
    assert c.val == {"a": 1, "b": 2}

    c2 = OmegaConf.create({"val": "${oc.dict_merge:null,invalid}"})
    assert c2.val == {}


def test_timestring():
    c = OmegaConf.create({"val": "${oc.timestring:}"})
    assert len(c.val) > 0


def test_oc_if():
    c1 = OmegaConf.create({"val": "${oc.if:True,yes,no}"})
    assert c1.val == "yes"
    c2 = OmegaConf.create({"val": "${oc.if:False,yes,no}"})
    assert c2.val == "no"
    c3 = OmegaConf.create({"val": "${oc.if:false,yes,no}"})
    assert c3.val == "no"
    c4 = OmegaConf.create({"val": "${oc.if:0,yes,no}"})
    assert c4.val == "no"


def test_len():
    c = OmegaConf.create({"val": "${oc.len:[1,2,3]}"})
    assert c.val == 3


def test_eval():
    c = OmegaConf.create({"val": "${oc.eval:1+1}"})
    assert c.val == 2


def test_eval_blocked_tokens():
    with pytest.raises(ValueError, match="blocked token"):
        OmegaConf.create({"val": "${oc.eval:'import os'}"}).val

    with pytest.raises(ValueError, match="blocked token"):
        OmegaConf.create({"val": "${oc.eval:'open(1)'}"}).val

    with pytest.raises(ValueError, match="blocked token"):
        OmegaConf.create({"val": "${oc.eval:'input(1)'}"}).val
