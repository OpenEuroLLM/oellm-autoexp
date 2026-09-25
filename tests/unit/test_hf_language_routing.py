import csv
import json
import math
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from oellm_autoexp.analysis.hf_language_routing import (
    GateAccumulator,
    ParallelCorpus,
    Qwen35RoutingRunner,
    TextRecord,
    analyze_parallel_corpus,
    distribution_summary,
    load_parallel_corpus,
    normalized_jsd,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class RoutingMetricTests(unittest.TestCase):
    def test_uniform_and_collapsed_distributions(self) -> None:
        uniform = distribution_summary([1, 1, 1, 1])
        collapsed = distribution_summary([4, 0, 0, 0])

        self.assertAlmostEqual(uniform["effective_experts"], 4.0)
        self.assertAlmostEqual(uniform["normalized_entropy"], 1.0)
        self.assertAlmostEqual(uniform["gini"], 0.0)
        self.assertAlmostEqual(collapsed["effective_experts"], 1.0)
        self.assertAlmostEqual(collapsed["normalized_entropy"], 0.0)
        self.assertAlmostEqual(collapsed["gini"], 0.75)
        self.assertEqual(collapsed["coverage_count"], 1)

    def test_normalized_jsd_bounds_and_symmetry(self) -> None:
        self.assertAlmostEqual(normalized_jsd([1, 0], [1, 0]), 0.0)
        self.assertAlmostEqual(normalized_jsd([1, 0], [0, 1]), 1.0)
        self.assertAlmostEqual(
            normalized_jsd([3, 1], [1, 3]),
            normalized_jsd([1, 3], [3, 1]),
        )

    def test_gate_summary(self) -> None:
        gate = GateAccumulator()
        gate.update(torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
        summary = gate.summary()

        self.assertEqual(summary["count"], 5)
        self.assertAlmostEqual(summary["mean"], 0.5)
        self.assertAlmostEqual(summary["fraction_above_0_5"], 0.4)
        self.assertAlmostEqual(summary["p50"], 0.5005, places=3)


class ParallelCorpusTests(unittest.TestCase):
    def test_join_uses_id_not_line_order_and_reports_data_quality(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            english = directory / "eng.jsonl"
            finnish = directory / "fin.jsonl"
            _write_jsonl(
                english,
                [
                    {"warc_record_id": "a", "text": "A", "partition": "train"},
                    {"warc_record_id": "b", "text": "B", "partition": "train"},
                    {"warc_record_id": "c", "text": "C", "partition": "test"},
                ],
            )
            _write_jsonl(
                finnish,
                [
                    {"warc_record_id": "c", "text": "C-fi", "partition": "dev"},
                    {"warc_record_id": "a", "text": "A-fi", "partition": "train"},
                ],
            )

            corpus = load_parallel_corpus(
                {"eng_Latn": english, "fin_Latn": finnish},
                "eng_Latn",
                max_pairs=0,
            )

        self.assertEqual(
            [record.record_id for record in corpus.records["fin_Latn"]],
            ["a", "c"],
        )
        self.assertEqual(corpus.missing_from_language["fin_Latn"], ["b"])
        self.assertEqual(corpus.partition_mismatches["fin_Latn"], 1)

    def test_sampling_is_deterministic(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            records = [
                {"warc_record_id": str(index), "text": f"text {index}"}
                for index in range(30)
            ]
            english = directory / "eng.jsonl"
            finnish = directory / "fin.jsonl"
            _write_jsonl(english, records)
            _write_jsonl(finnish, list(reversed(records)))
            files = {"eng_Latn": english, "fin_Latn": finnish}

            first = load_parallel_corpus(files, "eng_Latn", max_pairs=7, seed=9)
            second = load_parallel_corpus(files, "eng_Latn", max_pairs=7, seed=9)

        self.assertEqual(
            [record.record_id for record in first.records["eng_Latn"]],
            [record.record_id for record in second.records["eng_Latn"]],
        )
        self.assertEqual(first.anchor_records_seen, 30)

    def test_zero_matches_is_an_error(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            english = directory / "eng.jsonl"
            finnish = directory / "fin.jsonl"
            _write_jsonl(english, [{"warc_record_id": "a", "text": "A"}])
            _write_jsonl(finnish, [{"warc_record_id": "b", "text": "B"}])

            with self.assertRaisesRegex(ValueError, "No records"):
                load_parallel_corpus(
                    {"eng_Latn": english, "fin_Latn": finnish},
                    "eng_Latn",
                )

    def test_duplicate_ids_are_rejected(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            english = directory / "eng.jsonl"
            finnish = directory / "fin.jsonl"
            duplicated = [
                {"warc_record_id": "a", "text": "first"},
                {"warc_record_id": "a", "text": "second"},
            ]
            _write_jsonl(english, duplicated)
            _write_jsonl(finnish, [{"warc_record_id": "a", "text": "target"}])

            with self.assertRaisesRegex(ValueError, "Duplicate warc_record_id"):
                load_parallel_corpus(
                    {"eng_Latn": english, "fin_Latn": finnish},
                    "eng_Latn",
                )


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [int(value) for value in text.split()]


class _TinyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.shared_expert_gate = torch.nn.Linear(4, 1, bias=False)
        torch.nn.init.zeros_(self.mlp.shared_expert_gate.weight)


class _TinyQwenModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(
            model_type="qwen3_5_moe_text",
            num_hidden_layers=2,
            num_experts=4,
            num_experts_per_tok=2,
        )
        self.embedding = torch.nn.Embedding(20, 4)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_TinyLayer(), _TinyLayer()])

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, **_kwargs):
        hidden = self.embedding(input_ids)
        router_logits = []
        for layer in self.model.layers:
            layer.mlp.shared_expert_gate(hidden.reshape(-1, 4))
            router_logits.append(
                torch.nn.functional.one_hot(input_ids.flatten() % 4, num_classes=4).float()
            )
        return types.SimpleNamespace(router_logits=tuple(router_logits))


class QwenRunnerTests(unittest.TestCase):
    def test_window_batch_padding_is_excluded_from_metrics(self) -> None:
        model = _TinyQwenModel()
        with Qwen35RoutingRunner(
            model,
            _TinyTokenizer(),
            max_length=3,
            batch_size=2,
        ) as runner:
            result = runner.analyze(TextRecord("id", "1 2 3 4 5", None, 1))

        self.assertEqual(result.token_count, 5)
        self.assertEqual(result.window_count, 2)
        for layer in result.layers:
            self.assertEqual(int(layer.selection_counts.sum().item()), 10)
            self.assertAlmostEqual(float(layer.probability_mass.sum().item()), 5.0, places=5)
            self.assertEqual(layer.gate.count, 5)
            self.assertAlmostEqual(layer.gate.summary()["mean"], 0.5)


class _ReportRunner:
    num_layers = 1
    num_experts = 2

    def analyze(self, record: TextRecord):
        from oellm_autoexp.analysis.hf_language_routing import (
            DocumentLayerStats,
            DocumentRoutingSummary,
        )

        # The unmatched English document deliberately has the opposite routing.
        first_expert = not (record.record_id == "unmatched")
        counts = torch.tensor([10.0, 0.0] if first_expert else [0.0, 10.0])
        layer = DocumentLayerStats(counts, counts.clone())
        layer.gate.update(torch.tensor([0.5]))
        return DocumentRoutingSummary(record.record_id, record.partition, 1, 1, [layer])


class ReportTests(unittest.TestCase):
    def test_comparison_uses_only_matched_anchor_documents(self) -> None:
        anchor_records = [
            TextRecord("matched", "secret anchor", "train", 1),
            TextRecord("unmatched", "secret unmatched", "train", 2),
        ]
        corpus = ParallelCorpus(
            anchor_language="eng_Latn",
            language_files={"eng_Latn": Path("eng"), "fin_Latn": Path("fin")},
            records={
                "eng_Latn": anchor_records,
                "fin_Latn": [TextRecord("matched", "secret target", "train", 1)],
            },
            missing_from_language={"fin_Latn": ["unmatched"]},
            partition_mismatches={"fin_Latn": 0},
            anchor_records_seen=2,
        )

        with TemporaryDirectory() as raw_directory:
            output_dir = Path(raw_directory)
            report = analyze_parallel_corpus(
                _ReportRunner(),
                corpus,
                model_metadata={"model": "tiny"},
                output_dir=output_dir,
                arguments={},
                make_plots=False,
            )
            serialized = (output_dir / "summary.json").read_text(encoding="utf-8")
            with (output_dir / "pair_metrics.csv").open(encoding="utf-8") as handle:
                pair_rows = list(csv.DictReader(handle))

        comparison = report["comparisons"]["fin_Latn"][0]
        self.assertEqual(comparison["matched_documents"], 1)
        self.assertAlmostEqual(comparison["selection_jsd"], 0.0)
        self.assertEqual(len(pair_rows), 1)
        self.assertNotIn("secret", serialized)
        self.assertFalse(math.isnan(report["languages"]["eng_Latn"][0]["shared_gate"]["mean"]))


if __name__ == "__main__":
    unittest.main()
