"""scripts/model_merging/merge_hf.py: weighted bf16 merge of HF safetensors
exports."""

import importlib.util
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "model_merging" / "merge_hf.py"
spec = importlib.util.spec_from_file_location("merge_hf", SCRIPT)
merge_hf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge_hf)


def _export(root, name, seed):
    g = torch.Generator().manual_seed(seed)
    d = root / name
    d.mkdir()
    # two shards, a bf16 matrix spanning more than one chunk, an fp32 vector
    st.save_file(
        {
            "a.weight": torch.randn(7, 5, generator=g).bfloat16(),
            "big.weight": torch.randn(3, 11, generator=g).bfloat16(),
        },
        str(d / "model-00001-of-00002.safetensors"),
    )
    st.save_file(
        {"norm.weight": torch.randn(9, generator=g)}, str(d / "model-00002-of-00002.safetensors")
    )
    (d / "config.json").write_text(json.dumps({"seed": seed}))
    return d


def _rne_bf16(x):
    """Independent reference: x (a python float) rounded to bf16, nearest-even (normal range)."""
    if x == 0:
        return x
    scale = 2.0 ** (math.frexp(x)[1] - 8)  # bf16 keeps 8 significant bits
    return round(x / scale) * scale  # round(): ties to even


def _chunk(xs, weights, dtype=torch.bfloat16):
    raws = [torch.tensor(x, dtype=dtype).view(torch.int16).numpy().tobytes() for x in xs]
    out = merge_hf.merge_chunk(raws, weights, "BF16")
    return torch.frombuffer(bytearray(out), dtype=torch.bfloat16).float().tolist()


def test_ties_go_to_even_and_cancellations_to_zero():
    # 1 and 1+2^-7 are adjacent bf16; their mean 1+2^-8 is an exact tie -> even (1.0);
    # (1+2^-7, 1+2^-6) -> 1+3*2^-8, tie -> even (1+2^-6)
    assert _chunk([[1.0, 1 + 2**-7], [1 + 2**-7, 1 + 2**-6]], [1, 1]) == [1.0, 1 + 2**-6]
    # a + (-a) + 0 == 0 exactly (fp32 per-term scaling left ~1e-11 here)
    a = 0.000560760498046875
    assert _chunk([[a], [-a], [0.0]], [1, 1, 1]) == [0.0]


@pytest.mark.parametrize(
    "spec_,expected",
    [
        ("uniform", [1, 1, 1]),
        ("linear", [1, 2, 3]),
        ("1,0,1", [1, 0, 1]),
    ],
)
def test_parse_weights(spec_, expected):
    assert merge_hf.parse_weights(spec_, 3) == pytest.approx(expected)


def test_parse_weights_rejects_bad():
    with pytest.raises(SystemExit):
        merge_hf.parse_weights("1,2", 3)
    with pytest.raises(SystemExit):
        merge_hf.parse_weights("1,-1,1", 3)


def test_merge_is_correctly_rounded(tmp_path, monkeypatch):
    monkeypatch.setattr(merge_hf, "CHUNK", 8)  # force several chunks per tensor
    names = [_export(tmp_path, f"run_{i}k", seed=i).name for i in range(3)]
    monkeypatch.setattr(
        "sys.argv",
        ["merge_hf.py", "--root", str(tmp_path), "--out", "m", "--weights", "linear", *names],
    )
    merge_hf.main()

    out = tmp_path / "m"
    assert (out / ".convert_done").exists() and not (tmp_path / "m.partial").exists()
    info = json.loads((out / "MERGE_INFO.json").read_text())
    assert info["weights"] == pytest.approx([1 / 6, 2 / 6, 3 / 6])
    assert info["update_scale"] == pytest.approx([5 / 6, 3 / 6])
    assert json.loads((out / "config.json").read_text()) == {"seed": 2}  # last input's files

    ins = [
        {
            **st.load_file(str(tmp_path / n / "model-00001-of-00002.safetensors")),
            **st.load_file(str(tmp_path / n / "model-00002-of-00002.safetensors")),
        }
        for n in names
    ]
    got = {
        **st.load_file(str(out / "model-00001-of-00002.safetensors")),
        **st.load_file(str(out / "model-00002-of-00002.safetensors")),
    }
    for k, v in got.items():
        assert v.dtype == ins[-1][k].dtype
        exact = (sum(w * t[k].double() for t, w in zip(ins, [1, 2, 3])) / 6).flatten().tolist()
        if v.dtype == torch.bfloat16:  # the correctly rounded mean, element for element
            assert v.flatten().float().tolist() == [_rne_bf16(x) for x in exact], k
        else:
            assert torch.equal(v.flatten(), torch.tensor(exact, dtype=torch.float64).float()), k

    # a second run on a finished merge is a no-op
    merge_hf.main()


def test_refuses_foreign_output_and_mismatched_inputs(tmp_path, monkeypatch):
    names = [_export(tmp_path, f"r{i}", seed=i).name for i in range(2)]
    (tmp_path / "taken").mkdir()  # exists without .convert_done: someone else's
    monkeypatch.setattr(
        "sys.argv", ["merge_hf.py", "--root", str(tmp_path), "--out", "taken", *names]
    )
    with pytest.raises(SystemExit):
        merge_hf.main()
    st.save_file(
        {"other.weight": torch.zeros(2).bfloat16()},
        str(tmp_path / names[0] / "model-00001-of-00002.safetensors"),
    )
    monkeypatch.setattr("sys.argv", ["merge_hf.py", "--root", str(tmp_path), "--out", "m", *names])
    with pytest.raises(SystemExit):
        merge_hf.main()
    assert not (tmp_path / "m").exists()


def test_leftover_partial_is_redone(tmp_path, monkeypatch):
    names = [_export(tmp_path, f"r{i}", seed=i).name for i in range(2)]
    (tmp_path / "m.partial").mkdir()
    (tmp_path / "m.partial" / "junk").write_text("x")
    monkeypatch.setattr("sys.argv", ["merge_hf.py", "--root", str(tmp_path), "--out", "m", *names])
    merge_hf.main()
    assert not (tmp_path / "m" / "junk").exists() and (tmp_path / "m" / ".convert_done").exists()
