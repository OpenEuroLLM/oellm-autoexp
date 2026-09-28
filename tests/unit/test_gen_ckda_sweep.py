"""scripts/korbi/gen_ckda_sweep.py: parameter counts, chain arithmetic, and that the
committed sweep files are what the spec generates."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "korbi" / "gen_ckda_sweep.py"
OUT_DIR = REPO / "config" / "experiments" / "korbi" / "complex_kda"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_ckda_sweep", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def spec(gen):
    return yaml.safe_load(gen.SPEC.read_text())


@pytest.fixture(scope="module")
def models(gen, spec):
    return gen.build_models(spec)


# From ComplexKDA lm_scaling/ladder_matched.json, which counts parameters by
# instantiating the real fla models on the meta device: (matched ffn, total N).
LADDER_MATCHED = {
    ("s47M", "attn"): (1536, 47_637_888),
    ("s47M", "kda_sig_lowrank"): (1472, 48_025_800),
    ("s47M", "kda_sig_hybrid_lowrank"): (1408, 47_265_270),
    ("s124M", "ckda_shipped_lowrank"): (2240, 125_451_234),
    ("s302M", "ckda_shipped_hybrid_lowrank"): (3456, 302_961_170),
    ("s588M", "kda_sig_lowrank"): (4992, 586_324_880),
    ("s983M", "ckda_shipped_hybrid_lowrank"): (5952, 984_364_080),
    ("s1p7B", "attn"): (8192, 1_713_735_680),
    ("s1p7B", "kda_sig_hybrid_lowrank"): (7936, 1_712_287_424),
}


@pytest.mark.parametrize("size,arm", sorted(LADDER_MATCHED))
def test_counts_match_instantiated_models(models, size, arm):
    assert (models[size][arm]["ffn"], models[size][arm]["params"]) == LADDER_MATCHED[(size, arm)]


def test_qknorm_adds_two_head_norms_per_layer(models, spec):
    for size, geom in spec["sizes"].items():
        extra = models[size]["attn_qknorm"]["params"] - models[size]["attn"]["params"]
        assert extra == 2 * spec["head_dim"] * geom["layers"]


def test_matched_width_is_the_closest_on_the_grid(gen, spec, models):
    """No neighbouring ffn on the grid lands closer to the dense target."""
    m = spec["ffn_multiple"]
    for size, geom in spec["sizes"].items():
        for arm, entry in models[size].items():
            layout, meg = spec["arms"][arm]["layout"], entry["megatron"]
            if layout["mixer"] == "attention":
                continue
            err = abs(entry["params"] - entry["target"])
            for ffn in (entry["ffn"] - m, entry["ffn"] + m):
                assert abs(gen.count_params(geom, ffn, layout, meg, spec) - entry["target"]) >= err


def test_pure_linear_stacks_have_no_attention_layer(gen, spec):
    for size, geom in spec["sizes"].items():
        meg = gen.arm_megatron(spec["arms"]["ckda_shipped_lowrank"], geom)
        freq = meg["linear_attention_freq"]
        assert not any((i + 1) % freq == 0 for i in range(geom["layers"]))


def test_chains_branch_at_eighty_percent(gen, spec, models):
    seq, df = spec["seq_len"], spec["decay_fraction"]
    for ladder, chains in spec["ladders"].items():
        for chain in chains:
            c = gen.plan_chain(chain, spec, models)
            for cell in c["cells"]:
                iters = math.ceil(cell["bt"] * 1e9 / (seq * chain["gbs"]))
                assert cell["iters"] == iters
                assert cell["start"] == int(iters * (1 - df))
            assert c["stable_iters"] == max(x["start"] for x in c["cells"])
            # one width per chain that divides the batch, at a micro-batch that fits
            dp = c["nodes"] * spec["gpus_per_node"]
            assert chain["gbs"] % dp == 0
            assert (chain["gbs"] // dp) % c["mbs"] == 0
            assert c["mbs"] <= spec["sizes"][chain["size"]]["mbs_cap"]
            assert c["fits"], f"{ladder} {chain}"


def test_finegrained_ladder_covers_small_budgets(spec):
    budgets = {(c["size"], b) for c in spec["ladders"]["ladder_finegrained"] for b in c["budgets"]}
    for size in ("s47M", "s124M", "s302M"):
        assert {(size, b) for b in (1, 1.5, 2, 4)} <= budgets


def test_committed_outputs_are_up_to_date(gen, tmp_path):
    """Editing the spec without regenerating (or the generated files by hand)
    leaves a sweep that does not say what the spec says."""
    gen.main(["--out-dir", str(tmp_path)])
    produced = sorted(p.name for p in tmp_path.iterdir())
    assert produced
    stale = [
        name for name in produced if (tmp_path / name).read_text() != (OUT_DIR / name).read_text()
    ]
    assert not stale, f"regenerate: .venv/bin/python {SCRIPT.relative_to(REPO)} ({stale})"
