"""scripts/korbi/flag_eval/lmeval_parts.py: part names, document partition, per-document metrics
(the merge itself runs lm-eval's aggregation and is checked on the cluster against an unsplit row)."""

from __future__ import annotations

import json

import pytest

from scripts.korbi.flag_eval import lmeval_parts as lp


def test_parse_and_keys():
    assert lp.parse("vllm/mmlu/5@1of3") == ("vllm/mmlu/5", 1, 3)
    assert lp.parse("vllm/mmlu/5") == ("vllm/mmlu/5", None, None)
    with pytest.raises(ValueError):
        lp.parse("vllm/mmlu/5@3of3")
    assert lp.part_key("mmlu", 5, 1, 3) == "mmlu_5@1of3"


class _Task:
    def __init__(self, n):
        self.eval_docs = list(range(n))


class _TM:
    def load(self, tasks):
        return {"tasks": {"g_b": _Task(5), "g_a": _Task(7)}, "groups": {}}


def test_part_samples_cover_every_document_once():
    parts = [lp.part_samples(_TM(), "g", k, 3) for k in range(3)]
    for leaf, n in (("g_a", 7), ("g_b", 5)):
        got = sorted(i for p in parts for i in p[leaf])
        assert got == list(range(n))
    assert parts[1]["g_a"] == [1, 4]


def test_raw_metrics_in_document_order(tmp_path):
    samples = {"t": [
        {"doc_id": 4, "filter": "none", "metrics": ["acc", "acc_norm"], "acc": 1.0, "acc_norm": 0.0},
        {"doc_id": 1, "filter": "none", "metrics": ["acc", "acc_norm"], "acc": 0.0, "acc_norm": 1.0},
    ]}
    raw = lp.raw_metrics(samples)
    assert sorted(map(tuple, ((m, f, tuple(v)) for m, f, v in raw["t"]))) == [
        ("acc", "none", (0.0, 1.0)), ("acc_norm", "none", (1.0, 0.0))]
    path = tmp_path / "p" / "part1of3.json"
    lp.write_part(path, "t", 0, 1, 3, samples)
    data = json.loads(path.read_text())
    assert data["part"] == [1, 3] and data["raw"] == raw
