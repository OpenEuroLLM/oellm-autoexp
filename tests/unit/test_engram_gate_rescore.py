"""The scale-invariant Engram gate rescore: gate math, alignment, and the offline report."""
from __future__ import annotations

import numpy as np
import pytest

from oellm_autoexp.hf_export.engram_gate import (
    align_to_targets,
    engram_layers,
    fixed_query_gate,
    fixed_query_logit,
    gate_from_logit,
    gate_summary,
    rescore,
    rescore_report,
    signed_sqrt,
    spearman,
)


def test_signed_sqrt_matches_formula():
    x = np.array([-4.0, -1e-9, 0.0, 1e-9, 0.25, 9.0])
    want = np.sign(x) * np.sqrt(np.maximum(np.abs(x), 1e-6))
    np.testing.assert_allclose(signed_sqrt(x), want, rtol=0, atol=1e-12)


def test_gate_from_logit_is_sigmoid_of_signed_sqrt():
    x = np.array([-4.0, -0.01, 0.0, 0.01, 4.0])
    want = 1.0 / (1.0 + np.exp(-signed_sqrt(x)))
    np.testing.assert_allclose(gate_from_logit(x), want, rtol=0, atol=1e-12)


def test_fixed_query_is_function_preserving_at_the_per_token_rms():
    # logit_scale = |h~| per token reproduces the current logit exactly.
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 4096)
    hn = rng.uniform(1e-4, 1.0, 4096)
    np.testing.assert_allclose(fixed_query_logit(x, hn, 1.0), x / hn, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fixed_query_gate(x, hn, 1.0), gate_from_logit(x / hn), rtol=0, atol=1e-12)


def test_fixed_query_is_scale_invariant_in_the_query():
    # Scaling the eps-limited query and its RMS by the same factor leaves the fixed gate unchanged:
    # that is the whole point of the eps-free norm.
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 0.01, 2048)
    hn = rng.uniform(1e-4, 0.1, 2048)
    for factor in (1e-4, 1.0, 1e4):
        np.testing.assert_allclose(
            fixed_query_gate(x * factor, hn * factor, 0.1),
            fixed_query_gate(x, hn, 0.1),
            rtol=0,
            atol=1e-12,
        )


def test_fixed_query_opens_the_gate_as_the_scale_grows():
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 0.02, 20000)
    hn = rng.uniform(1e-4, 0.1, 20000)
    spreads = []
    for scale in (0.01, 0.1, 1.0):
        a = fixed_query_gate(x, hn, scale)
        spreads.append(np.percentile(a, 90) - np.percentile(a, 10))
    assert spreads[0] < spreads[1] < spreads[2]


def test_align_to_targets_drops_the_last_token_of_each_document():
    values = np.array([10, 11, 12, 20, 21, 22, 23])  # docs of 2 and 3 tokens, BOS first
    out = align_to_targets(values, [2, 3])
    np.testing.assert_array_equal(out, [10, 11, 20, 21, 22])


def test_align_to_targets_empty():
    assert align_to_targets(np.array([]), []).size == 0


def test_spearman_perfect_and_inverse():
    a = np.arange(100.0)
    assert spearman(a, a) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)
    assert np.isnan(spearman(a, np.ones_like(a)))
    assert np.isnan(spearman(a[:2], a[:2]))


def test_gate_summary_separates_top_and_bottom_deciles():
    alpha = np.linspace(0.1, 0.9, 1000)
    utility = alpha.copy()
    s = gate_summary(alpha, utility)
    assert s["rho"] == pytest.approx(1.0)
    assert s["utility_top_decile"] > s["utility_bottom_decile"]
    assert s["gate_p10"] < s["gate_p50"] < s["gate_p90"]


def _synthetic_npz(path, layers=(2, 5), doc_lengths=(3, 2), perfect=True, constant=False):
    doc_lengths = np.asarray(doc_lengths, dtype=np.int64)
    n_gate = int(doc_lengths.sum() + len(doc_lengths))
    n_nll = int(doc_lengths.sum())
    arrays = {"doc_lengths": doc_lengths}
    for i in layers:
        if constant:
            x = np.zeros(n_gate, dtype=np.float32)
        else:
            # aligned positions (0..2 and 4..5) rise; utility rises with them.
            x = np.zeros(n_gate, dtype=np.float32)
            x[[0, 1, 2, 4, 5]] = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            x[[3, 6]] = 0.35
        hn = np.linspace(0.01, 0.02, n_gate).astype(np.float32)
        arrays[f"x_L{i}"] = x
        arrays[f"hn_L{i}"] = hn
        arrays[f"alpha_L{i}"] = gate_from_logit(x).astype(np.float32)
        utility = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32) if perfect else np.zeros(n_nll, np.float32)
        arrays[f"zero_L{i}"] = utility
    arrays["base"] = np.zeros(n_nll, dtype=np.float32)
    np.savez(path, **arrays)
    return np.load(path)


def test_engram_layers_sorted(tmp_path):
    z = _synthetic_npz(tmp_path / "a.npz", layers=(5, 2))
    assert engram_layers(z) == [2, 5]


def test_rescore_synthetic_perfect_correlation(tmp_path):
    z = _synthetic_npz(tmp_path / "a.npz")
    res = rescore(z, [0.025, 0.1, 1.0], sweep_layer=2)
    assert res["layers"] == [2, 5]
    assert res["sweep_layer"] == 2
    for i in (2, 5):
        assert res["detail"][i]["current"]["rho"] == pytest.approx(1.0)
        for scale, s in res["detail"][i]["fixed"].items():
            assert s["rho"] == pytest.approx(1.0), (i, scale)
    # the non-swept layer keeps its mean-|h~| init, so it stays close to the current gate
    assert res["detail"][5]["fixed"][1.0]["logit_scale"] == pytest.approx(res["hn_mean"][5])


def test_rescore_constant_gate_has_no_rank_correlation(tmp_path):
    z = _synthetic_npz(tmp_path / "a.npz", constant=True)
    res = rescore(z, [0.025, 1.0], sweep_layer=2)
    for i in (2, 5):
        assert np.isnan(res["detail"][i]["current"]["rho"])
        for s in res["detail"][i]["fixed"].values():
            assert np.isnan(s["rho"])


def test_rescore_report_lists_every_scale(tmp_path):
    z = _synthetic_npz(tmp_path / "a.npz")
    res = rescore(z, [0.025, 0.1, 0.3, 1.0], sweep_layer=2)
    report = rescore_report(res, source="synthetic")
    assert "synthetic" in report
    assert "## Engram layer 2" in report and "## Engram layer 5" in report
    for scale in ("0.025", "0.1", "0.3", "1"):
        assert f"| {scale} |" in report
    assert "current" in report


def test_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 0.5, 4096)
    hn = rng.uniform(1e-4, 1.0, 4096)
    xt = torch.tensor(x, dtype=torch.float32)
    hnt = torch.tensor(hn, dtype=torch.float32)
    xf = xt * 0.1 / hnt.clamp_min(1e-12)
    got = torch.sigmoid(xf.abs().clamp_min(1e-6).sqrt() * xf.sign()).numpy()
    want = fixed_query_gate(x, hn, 0.1)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)


def test_rescore_reports_nll_jump_only_on_the_swept_layer(tmp_path):
    z = _synthetic_npz(tmp_path / "a.npz")
    arrays = {k: z[k] for k in z.files}
    arrays["fixed_ls0.1"] = arrays["base"] + np.float32(0.02)
    np.savez(tmp_path / "b.npz", **arrays)
    res = rescore(np.load(tmp_path / "b.npz"), [0.1], sweep_layer=2)
    assert res["detail"][2]["fixed"][0.1]["nll_jump"] == pytest.approx(0.02, abs=1e-6)
    assert "nll_jump" not in res["detail"][5]["fixed"][0.1]
    report = rescore_report(res, source="synthetic")
    assert "+0.0200" in report
