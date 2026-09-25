"""Language-conditioned routing analysis for Qwen3.5 MoE Hugging Face models.

The implementation intentionally supports only Qwen3.5 MoE.  Hugging Face MoE
models do not expose a common router/shared-expert interface, and silently
guessing module semantics would make the resulting research metrics unreliable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch


TOP_N_VALUES = (1, 4, 8, 16, 32)
GATE_HISTOGRAM_BINS = 1000
SUPPORTED_MODEL_TYPES = {"qwen3_5_moe", "qwen3_5_moe_text"}


@dataclass(frozen=True)
class TextRecord:
    record_id: str
    text: str
    partition: str | None
    line_number: int


@dataclass
class ParallelCorpus:
    anchor_language: str
    language_files: dict[str, Path]
    records: dict[str, list[TextRecord]]
    missing_from_language: dict[str, list[str]]
    partition_mismatches: dict[str, int]
    anchor_records_seen: int


@dataclass
class GateAccumulator:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    above_half: int = 0
    histogram: list[int] = field(
        default_factory=lambda: [0 for _ in range(GATE_HISTOGRAM_BINS)]
    )

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().to(dtype=torch.float32, device="cpu").flatten()
        if values.numel() == 0:
            return
        self.count += int(values.numel())
        self.total += float(values.sum().item())
        self.total_sq += float(values.square().sum().item())
        self.above_half += int((values > 0.5).sum().item())
        indices = (values.clamp(0.0, 1.0) * GATE_HISTOGRAM_BINS).to(torch.long)
        indices.clamp_(max=GATE_HISTOGRAM_BINS - 1)
        counts = torch.bincount(indices, minlength=GATE_HISTOGRAM_BINS)
        for index, count in enumerate(counts.tolist()):
            self.histogram[index] += int(count)

    def merge(self, other: "GateAccumulator") -> None:
        self.count += other.count
        self.total += other.total
        self.total_sq += other.total_sq
        self.above_half += other.above_half
        self.histogram = [a + b for a, b in zip(self.histogram, other.histogram)]

    def summary(self) -> dict[str, float | int]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": float("nan"),
                "std": float("nan"),
                "p05": float("nan"),
                "p50": float("nan"),
                "p95": float("nan"),
                "fraction_above_0_5": float("nan"),
            }
        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "p05": histogram_quantile(self.histogram, 0.05),
            "p50": histogram_quantile(self.histogram, 0.50),
            "p95": histogram_quantile(self.histogram, 0.95),
            "fraction_above_0_5": self.above_half / self.count,
        }


@dataclass
class DocumentLayerStats:
    selection_counts: torch.Tensor
    probability_mass: torch.Tensor
    gate: GateAccumulator = field(default_factory=GateAccumulator)

    @classmethod
    def empty(cls, num_experts: int) -> "DocumentLayerStats":
        return cls(
            selection_counts=torch.zeros(num_experts, dtype=torch.float64),
            probability_mass=torch.zeros(num_experts, dtype=torch.float64),
        )


@dataclass
class DocumentRoutingSummary:
    record_id: str
    partition: str | None
    token_count: int
    window_count: int
    layers: list[DocumentLayerStats]


@dataclass
class DocumentLayerReference:
    """Compact anchor statistics retained until translated documents are analyzed."""

    selection_counts: torch.Tensor
    probability_mass: torch.Tensor
    shared_gate_mean: float


@dataclass
class DocumentRoutingReference:
    record_id: str
    partition: str | None
    token_count: int
    layers: list[DocumentLayerReference]


def compact_document_reference(
    document: DocumentRoutingSummary,
) -> DocumentRoutingReference:
    return DocumentRoutingReference(
        record_id=document.record_id,
        partition=document.partition,
        token_count=document.token_count,
        layers=[
            DocumentLayerReference(
                selection_counts=layer.selection_counts.to(dtype=torch.float32),
                probability_mass=layer.probability_mass.to(dtype=torch.float32),
                shared_gate_mean=float(layer.gate.summary()["mean"]),
            )
            for layer in document.layers
        ],
    )


@dataclass
class LanguageLayerAccumulator:
    selection_counts: torch.Tensor
    probability_mass: torch.Tensor
    document_presence: torch.Tensor
    top_set_frequency: torch.Tensor
    gate: GateAccumulator = field(default_factory=GateAccumulator)

    @classmethod
    def empty(cls, num_experts: int) -> "LanguageLayerAccumulator":
        return cls(
            selection_counts=torch.zeros(num_experts, dtype=torch.float64),
            probability_mass=torch.zeros(num_experts, dtype=torch.float64),
            document_presence=torch.zeros(num_experts, dtype=torch.int64),
            top_set_frequency=torch.zeros(num_experts, dtype=torch.int64),
        )


@dataclass
class LanguageAccumulator:
    layers: list[LanguageLayerAccumulator]
    documents: int = 0
    tokens: int = 0
    windows: int = 0

    @classmethod
    def empty(cls, num_layers: int, num_experts: int) -> "LanguageAccumulator":
        return cls(
            layers=[LanguageLayerAccumulator.empty(num_experts) for _ in range(num_layers)]
        )

    def add_document(self, document: DocumentRoutingSummary, top_set_size: int = 16) -> None:
        self.documents += 1
        self.tokens += document.token_count
        self.windows += document.window_count
        for aggregate, layer in zip(self.layers, document.layers):
            aggregate.selection_counts += layer.selection_counts
            aggregate.probability_mass += layer.probability_mass
            aggregate.document_presence += layer.selection_counts.gt(0).to(torch.int64)
            n = min(top_set_size, layer.selection_counts.numel())
            if n and layer.selection_counts.sum().item() > 0:
                indices = torch.topk(layer.selection_counts, k=n).indices
                aggregate.top_set_frequency[indices] += 1
            aggregate.gate.merge(layer.gate)


def parse_language_specs(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --language value {value!r}; expected CODE=PATH")
        language, raw_path = value.split("=", 1)
        language = language.strip()
        path = Path(raw_path).expanduser()
        if not language or not raw_path:
            raise ValueError(f"Invalid --language value {value!r}; expected CODE=PATH")
        if language in result:
            raise ValueError(f"Duplicate language code: {language}")
        if not path.is_file():
            raise FileNotFoundError(f"Language file does not exist: {path}")
        result[language] = path
    if len(result) < 2:
        raise ValueError("Provide at least two --language CODE=PATH inputs")
    return result


def _parse_jsonl_record(
    line: str,
    *,
    path: Path,
    line_number: int,
    id_field: str,
    text_field: str,
    partition_field: str,
) -> TextRecord:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON at {path}:{line_number}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}:{line_number}")
    if id_field not in value:
        raise ValueError(f"Missing {id_field!r} at {path}:{line_number}")
    if text_field not in value:
        raise ValueError(f"Missing {text_field!r} at {path}:{line_number}")
    record_id = str(value[id_field]).strip()
    text = value[text_field]
    if not record_id:
        raise ValueError(f"Empty {id_field!r} at {path}:{line_number}")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Empty/non-string {text_field!r} at {path}:{line_number}")
    raw_partition = value.get(partition_field)
    partition = None if raw_partition is None else str(raw_partition)
    return TextRecord(record_id, text, partition, line_number)


def iter_jsonl_records(
    path: Path,
    *,
    id_field: str = "warc_record_id",
    text_field: str = "text",
    partition_field: str = "partition",
) -> Iterator[TextRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            yield _parse_jsonl_record(
                line,
                path=path,
                line_number=line_number,
                id_field=id_field,
                text_field=text_field,
                partition_field=partition_field,
            )


def _sample_anchor_records(
    records: Iterable[TextRecord], max_pairs: int, seed: int
) -> tuple[list[TextRecord], int]:
    rng = random.Random(seed)
    reservoir: list[TextRecord] = []
    seen_ids: set[str] = set()
    count = 0
    for record in records:
        if record.record_id in seen_ids:
            raise ValueError(f"Duplicate warc_record_id in anchor file: {record.record_id}")
        seen_ids.add(record.record_id)
        count += 1
        if max_pairs == 0:
            reservoir.append(record)
        elif len(reservoir) < max_pairs:
            reservoir.append(record)
        else:
            replacement = rng.randrange(count)
            if replacement < max_pairs:
                reservoir[replacement] = record
    reservoir.sort(key=lambda item: item.line_number)
    return reservoir, count


def load_parallel_corpus(
    language_files: Mapping[str, Path],
    anchor_language: str,
    *,
    max_pairs: int = 1000,
    seed: int = 1234,
    id_field: str = "warc_record_id",
    text_field: str = "text",
    partition_field: str = "partition",
) -> ParallelCorpus:
    if anchor_language not in language_files:
        raise ValueError(f"Anchor language {anchor_language!r} was not provided")
    if max_pairs < 0:
        raise ValueError("--max-pairs must be >= 0")

    anchor_records, anchor_seen = _sample_anchor_records(
        iter_jsonl_records(
            language_files[anchor_language],
            id_field=id_field,
            text_field=text_field,
            partition_field=partition_field,
        ),
        max_pairs,
        seed,
    )
    if not anchor_records:
        raise ValueError("Anchor language file contains no valid records")
    anchor_by_id = {record.record_id: record for record in anchor_records}
    selected_ids = set(anchor_by_id)

    records: dict[str, list[TextRecord]] = {anchor_language: anchor_records}
    missing: dict[str, list[str]] = {}
    partition_mismatches: dict[str, int] = {}

    for language, path in language_files.items():
        if language == anchor_language:
            continue
        matched: dict[str, TextRecord] = {}
        seen_ids: set[str] = set()
        for record in iter_jsonl_records(
            path,
            id_field=id_field,
            text_field=text_field,
            partition_field=partition_field,
        ):
            if record.record_id in seen_ids:
                raise ValueError(f"Duplicate warc_record_id in {language}: {record.record_id}")
            seen_ids.add(record.record_id)
            if record.record_id in selected_ids:
                matched[record.record_id] = record
        records[language] = [
            matched[item.record_id]
            for item in anchor_records
            if item.record_id in matched
        ]
        missing[language] = [
            item.record_id for item in anchor_records if item.record_id not in matched
        ]
        partition_mismatches[language] = sum(
            1
            for record_id, target in matched.items()
            if anchor_by_id[record_id].partition != target.partition
        )
        if not records[language]:
            raise ValueError(
                f"No records from {language!r} match the sampled {anchor_language!r} "
                f"records by {id_field!r}"
            )

    return ParallelCorpus(
        anchor_language=anchor_language,
        language_files=dict(language_files),
        records=records,
        missing_from_language=missing,
        partition_mismatches=partition_mismatches,
        anchor_records_seen=anchor_seen,
    )


def normalize_distribution(values: Sequence[float] | torch.Tensor) -> list[float]:
    if isinstance(values, torch.Tensor):
        values = values.detach().to(dtype=torch.float64, device="cpu").tolist()
    clean = [max(float(value), 0.0) for value in values]
    total = sum(clean)
    if total <= 0:
        return [0.0 for _ in clean]
    return [value / total for value in clean]


def entropy(distribution: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in distribution if value > 0)


def gini(values: Sequence[float] | torch.Tensor) -> float:
    if isinstance(values, torch.Tensor):
        values = values.detach().to(dtype=torch.float64, device="cpu").tolist()
    clean = sorted(max(float(value), 0.0) for value in values)
    total = sum(clean)
    count = len(clean)
    if count == 0 or total == 0:
        return 0.0
    weighted = sum((2 * index - count - 1) * value for index, value in enumerate(clean, 1))
    return weighted / (count * total)


def normalized_jsd(
    left: Sequence[float] | torch.Tensor, right: Sequence[float] | torch.Tensor
) -> float:
    p = normalize_distribution(left)
    q = normalize_distribution(right)
    if not p or len(p) != len(q):
        raise ValueError("JSD inputs must be non-empty and have equal length")
    if not any(p) and not any(q):
        return 0.0
    midpoint = [(a + b) / 2 for a, b in zip(p, q)]

    def kl_divergence(first: Sequence[float], second: Sequence[float]) -> float:
        return sum(a * math.log(a / b) for a, b in zip(first, second) if a > 0 and b > 0)

    return (kl_divergence(p, midpoint) + kl_divergence(q, midpoint)) / (2 * math.log(2))


def histogram_quantile(histogram: Sequence[int], quantile: float) -> float:
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    total = sum(histogram)
    if total == 0:
        return float("nan")
    target = max(1, math.ceil(quantile * total))
    cumulative = 0
    for index, count in enumerate(histogram):
        cumulative += count
        if cumulative >= target:
            return (index + 0.5) / len(histogram)
    return 1.0


def distribution_summary(values: Sequence[float] | torch.Tensor) -> dict[str, float | int]:
    distribution = normalize_distribution(values)
    count = len(distribution)
    ent = entropy(distribution)
    sorted_values = sorted(distribution, reverse=True)
    result: dict[str, float | int] = {
        "entropy": ent,
        "normalized_entropy": ent / math.log(count) if count > 1 else 0.0,
        "effective_experts": math.exp(ent),
        "gini": gini(distribution),
        "coverage_count": sum(value > 0 for value in distribution),
        "coverage_fraction": sum(value > 0 for value in distribution) / count if count else 0.0,
    }
    for n in TOP_N_VALUES:
        result[f"top_{n}_mass"] = sum(sorted_values[: min(n, count)])
    return result


def top_set(values: torch.Tensor, size: int = 16) -> set[int]:
    if values.numel() == 0 or values.sum().item() <= 0:
        return set()
    return set(torch.topk(values, k=min(size, values.numel())).indices.tolist())


def jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


class Qwen35RoutingRunner:
    """Run Qwen3.5 forwards and aggregate router information online."""

    _SHARED_GATE_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.shared_expert_gate$")

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_length: int,
        batch_size: int,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        config = getattr(model, "config", None)
        model_type = getattr(config, "model_type", None)
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type {model_type!r}; expected Qwen3.5 MoE "
                f"({', '.join(sorted(SUPPORTED_MODEL_TYPES))})"
            )
        text_config = getattr(config, "text_config", config)
        self.num_layers = int(text_config.num_hidden_layers)
        self.num_experts = int(text_config.num_experts)
        self.top_k = int(text_config.num_experts_per_tok)
        self._gate_outputs: dict[int, torch.Tensor] = {}
        self._hook_handles: list[Any] = []
        self._register_shared_gate_hooks()

        embedding = model.get_input_embeddings()
        self.input_device = embedding.weight.device
        if self.input_device.type == "meta":
            raise RuntimeError("Model input embeddings remain on the meta device")

    def _register_shared_gate_hooks(self) -> None:
        found: dict[int, Any] = {}
        for name, module in self.model.named_modules():
            match = self._SHARED_GATE_RE.search(name)
            if match:
                layer_index = int(match.group(1))
                if layer_index in found:
                    raise RuntimeError(
                        f"Found multiple shared-expert gates for text layer {layer_index}"
                    )
                found[layer_index] = module
        expected = set(range(self.num_layers))
        if set(found) != expected:
            missing = sorted(expected - set(found))
            extra = sorted(set(found) - expected)
            raise RuntimeError(
                f"Could not discover exactly one shared gate per text layer; "
                f"missing={missing}, extra={extra}"
            )

        for layer_index, module in sorted(found.items()):

            def hook(_module, _inputs, output, *, index=layer_index):
                self._gate_outputs[index] = torch.sigmoid(
                    output.detach().to(dtype=torch.float32, device="cpu").flatten()
                )

            self._hook_handles.append(module.register_forward_hook(hook))

    def close(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def __enter__(self) -> "Qwen35RoutingRunner":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def _token_windows(self, text: str) -> list[list[int]]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            raise ValueError("Tokenizer produced no tokens for a non-empty record")
        return [
            token_ids[start : start + self.max_length]
            for start in range(0, len(token_ids), self.max_length)
        ]

    def analyze(self, record: TextRecord) -> DocumentRoutingSummary:
        windows = self._token_windows(record.text)
        layers = [DocumentLayerStats.empty(self.num_experts) for _ in range(self.num_layers)]
        token_count = sum(len(window) for window in windows)

        for start in range(0, len(windows), self.batch_size):
            chunk = windows[start : start + self.batch_size]
            max_width = max(len(item) for item in chunk)
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id
            if pad_id is None:
                raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
            input_ids = torch.full((len(chunk), max_width), int(pad_id), dtype=torch.long)
            attention_mask = torch.zeros((len(chunk), max_width), dtype=torch.long)
            for row, values in enumerate(chunk):
                input_ids[row, : len(values)] = torch.tensor(values, dtype=torch.long)
                attention_mask[row, : len(values)] = 1

            self._gate_outputs.clear()
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids.to(self.input_device),
                    attention_mask=attention_mask.to(self.input_device),
                    use_cache=False,
                    output_router_logits=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
            router_logits = getattr(outputs, "router_logits", None)
            if router_logits is None or len(router_logits) != self.num_layers:
                observed = None if router_logits is None else len(router_logits)
                raise RuntimeError(
                    f"Expected router logits for {self.num_layers} layers, observed {observed}"
                )
            if set(self._gate_outputs) != set(range(self.num_layers)):
                raise RuntimeError("Shared-expert gate hooks did not run for every text layer")

            flat_mask = attention_mask.bool().flatten()
            for layer_index, raw_logits in enumerate(router_logits):
                logits = raw_logits.detach().reshape(-1, self.num_experts)
                mask = flat_mask.to(logits.device)
                logits = logits[mask]
                probabilities = torch.softmax(logits.float(), dim=-1)
                selected = torch.topk(probabilities, k=self.top_k, dim=-1).indices
                counts = torch.bincount(
                    selected.flatten(), minlength=self.num_experts
                ).to(dtype=torch.float64, device="cpu")
                layers[layer_index].selection_counts += counts
                layers[layer_index].probability_mass += probabilities.sum(dim=0).to(
                    dtype=torch.float64, device="cpu"
                )
                gate_values = self._gate_outputs[layer_index]
                layers[layer_index].gate.update(gate_values[flat_mask])
            del outputs

        return DocumentRoutingSummary(
            record_id=record.record_id,
            partition=record.partition,
            token_count=token_count,
            window_count=len(windows),
            layers=layers,
        )


def load_qwen35_runner(
    model_name: str,
    *,
    tokenizer_name: str | None,
    revision: str | None,
    local_files_only: bool,
    dtype_name: str,
    device_map: str,
    max_length: int,
    batch_size: int,
) -> tuple[Qwen35RoutingRunner, dict[str, Any]]:
    try:
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install the analysis dependencies with `pip install -e '.[analysis]'`"
        ) from exc

    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_by_name:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    common = {
        "revision": revision,
        "local_files_only": local_files_only,
    }
    config = AutoConfig.from_pretrained(model_name, **common)
    if config.model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type {config.model_type!r}; this tool supports Qwen3.5 MoE only"
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=dtype_by_name[dtype_name],
        device_map=device_map,
        low_cpu_mem_usage=True,
        **common,
    )
    model.eval()
    tokenizer_source = tokenizer_name or model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **common)
    runner = Qwen35RoutingRunner(
        model,
        tokenizer,
        max_length=max_length,
        batch_size=batch_size,
    )
    metadata = {
        "model": model_name,
        "tokenizer": tokenizer_source,
        "revision": revision,
        "model_type": config.model_type,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "dtype": dtype_name,
        "device_map": device_map,
        "num_layers": runner.num_layers,
        "num_experts": runner.num_experts,
        "top_k": runner.top_k,
    }
    return runner, metadata


def layer_summary(layer: LanguageLayerAccumulator, documents: int) -> dict[str, Any]:
    selection = distribution_summary(layer.selection_counts)
    probability = distribution_summary(layer.probability_mass)
    top_indices = torch.topk(
        layer.selection_counts, k=min(16, layer.selection_counts.numel())
    ).indices
    if documents:
        persistence = float(layer.top_set_frequency[top_indices].double().mean().item()) / documents
        max_document_frequency = float(layer.document_presence.max().item()) / documents
    else:
        persistence = float("nan")
        max_document_frequency = float("nan")
    return {
        "selection": selection,
        "probability_mass": probability,
        "top_16_document_persistence": persistence,
        "max_expert_document_frequency": max_document_frequency,
        "shared_gate": layer.gate.summary(),
    }


def _flatten_layer_row(
    *, language: str, layer_index: int, aggregate: LanguageLayerAccumulator, documents: int
) -> dict[str, Any]:
    summary = layer_summary(aggregate, documents)
    row: dict[str, Any] = {
        "row_type": "language",
        "language": language,
        "layer": layer_index,
        "documents": documents,
        "top_16_document_persistence": summary["top_16_document_persistence"],
        "max_expert_document_frequency": summary["max_expert_document_frequency"],
    }
    for family in ("selection", "probability_mass"):
        for key, value in summary[family].items():
            row[f"{family}_{key}"] = value
    for key, value in summary["shared_gate"].items():
        row[f"shared_gate_{key}"] = value
    return row


def _document_pair_rows(
    anchor_language: str,
    target_language: str,
    anchor: DocumentRoutingReference,
    target: DocumentRoutingSummary,
) -> list[dict[str, Any]]:
    rows = []
    for layer_index, (left, right) in enumerate(zip(anchor.layers, target.layers)):
        right_gate = right.gate.summary()
        rows.append(
            {
                "warc_record_id": anchor.record_id,
                "partition": anchor.partition,
                "anchor_language": anchor_language,
                "target_language": target_language,
                "layer": layer_index,
                "anchor_tokens": anchor.token_count,
                "target_tokens": target.token_count,
                "selection_jsd": normalized_jsd(
                    left.selection_counts, right.selection_counts
                ),
                "probability_mass_jsd": normalized_jsd(
                    left.probability_mass, right.probability_mass
                ),
                "top_16_jaccard": jaccard(
                    top_set(left.selection_counts), top_set(right.selection_counts)
                ),
                "anchor_shared_gate_mean": left.shared_gate_mean,
                "target_shared_gate_mean": right_gate["mean"],
                "shared_gate_mean_delta": (
                    float(right_gate["mean"]) - left.shared_gate_mean
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _make_plots(
    output_dir: Path,
    language_layers: Mapping[str, Sequence[Mapping[str, Any]]],
    comparisons: Mapping[str, Sequence[Mapping[str, Any]]],
    expert_rows: Sequence[Mapping[str, Any]],
    anchor_language: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Matplotlib is required for plot output") from exc

    layers = range(len(next(iter(language_layers.values()))))

    def line_plot(filename: str, selector, ylabel: str) -> None:
        fig, axis = plt.subplots(figsize=(9, 5))
        for language, values in language_layers.items():
            axis.plot(list(layers), [selector(item) for item in values], label=language)
        axis.set_xlabel("Layer")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    line_plot(
        "effective_experts_by_layer.png",
        lambda item: item["selection"]["effective_experts"],
        "Effective selected experts",
    )
    line_plot(
        "routing_concentration_by_layer.png",
        lambda item: item["selection"]["gini"],
        "Selection Gini",
    )
    line_plot(
        "shared_gate_by_layer.png",
        lambda item: item["shared_gate"]["mean"],
        "Mean shared-expert gate",
    )

    for target, values in comparisons.items():
        fig, axis = plt.subplots(figsize=(9, 5))
        axis.plot(list(layers), [item["selection_jsd"] for item in values], label="selection")
        axis.plot(
            list(layers),
            [item["probability_mass_jsd"] for item in values],
            label="probability mass",
        )
        axis.set_xlabel("Layer")
        axis.set_ylabel("Normalized JSD")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"jsd_{anchor_language}_{target}.png", dpi=180)
        plt.close(fig)

    by_language: dict[str, torch.Tensor] = {}
    languages = list(language_layers)
    num_layers = len(next(iter(language_layers.values())))
    num_experts = max(int(row["expert"]) for row in expert_rows) + 1
    for language in languages:
        matrix = torch.zeros((num_layers, num_experts), dtype=torch.float64)
        for row in expert_rows:
            if row["language"] == language:
                matrix[int(row["layer"]), int(row["expert"])] = float(row["selection_fraction"])
        by_language[language] = matrix
        fig, axis = plt.subplots(figsize=(12, 5))
        image = axis.imshow(matrix.numpy(), aspect="auto", interpolation="nearest")
        axis.set_xlabel("Expert")
        axis.set_ylabel("Layer")
        axis.set_title(language)
        fig.colorbar(image, ax=axis, label="Selection fraction")
        fig.tight_layout()
        fig.savefig(output_dir / f"expert_routing_heatmap_{language}.png", dpi=180)
        plt.close(fig)
    for target in languages:
        if target == anchor_language:
            continue
        difference = by_language[target] - by_language[anchor_language]
        bound = max(float(difference.abs().max().item()), 1e-12)
        fig, axis = plt.subplots(figsize=(12, 5))
        image = axis.imshow(
            difference.numpy(),
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-bound,
            vmax=bound,
        )
        axis.set_xlabel("Expert")
        axis.set_ylabel("Layer")
        axis.set_title(f"{target} - {anchor_language}")
        fig.colorbar(image, ax=axis, label="Selection-fraction delta")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"expert_routing_difference_{anchor_language}_{target}.png",
            dpi=180,
        )
        plt.close(fig)


def analyze_parallel_corpus(
    runner: Qwen35RoutingRunner,
    corpus: ParallelCorpus,
    *,
    model_metadata: Mapping[str, Any],
    output_dir: Path,
    arguments: Mapping[str, Any],
    make_plots: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    language_accumulators = {
        language: LanguageAccumulator.empty(runner.num_layers, runner.num_experts)
        for language in corpus.records
    }

    anchor_documents: dict[str, DocumentRoutingReference] = {}
    anchor_language = corpus.anchor_language
    anchor_records = corpus.records[anchor_language]
    target_record_ids = {
        language: {record.record_id for record in records}
        for language, records in corpus.records.items()
        if language != anchor_language
    }
    matched_anchor_accumulators = {
        language: LanguageAccumulator.empty(runner.num_layers, runner.num_experts)
        for language in target_record_ids
    }
    print(f"Analyzing {len(anchor_records)} {anchor_language} anchor documents", flush=True)
    for index, record in enumerate(anchor_records, start=1):
        summary = runner.analyze(record)
        language_accumulators[anchor_language].add_document(summary)
        for language, record_ids in target_record_ids.items():
            if record.record_id in record_ids:
                matched_anchor_accumulators[language].add_document(summary)
        anchor_documents[record.record_id] = compact_document_reference(summary)
        if index % 25 == 0 or index == len(anchor_records):
            print(f"  {anchor_language}: {index}/{len(anchor_records)}", flush=True)

    pair_rows: list[dict[str, Any]] = []
    for language, records in corpus.records.items():
        if language == anchor_language:
            continue
        print(f"Analyzing {len(records)} matched {language} documents", flush=True)
        for index, record in enumerate(records, start=1):
            summary = runner.analyze(record)
            language_accumulators[language].add_document(summary)
            anchor_summary = anchor_documents[record.record_id]
            pair_rows.extend(
                _document_pair_rows(
                    anchor_language,
                    language,
                    anchor_summary,
                    summary,
                )
            )
            if index % 25 == 0 or index == len(records):
                print(f"  {language}: {index}/{len(records)}", flush=True)

    language_layers: dict[str, list[dict[str, Any]]] = {}
    layer_rows: list[dict[str, Any]] = []
    expert_rows: list[dict[str, Any]] = []
    for language, accumulator in language_accumulators.items():
        language_layers[language] = []
        for layer_index, layer in enumerate(accumulator.layers):
            summary = layer_summary(layer, accumulator.documents)
            language_layers[language].append(summary)
            layer_rows.append(
                _flatten_layer_row(
                    language=language,
                    layer_index=layer_index,
                    aggregate=layer,
                    documents=accumulator.documents,
                )
            )
            selection_distribution = normalize_distribution(layer.selection_counts)
            probability_distribution = normalize_distribution(layer.probability_mass)
            for expert in range(runner.num_experts):
                expert_rows.append(
                    {
                        "language": language,
                        "layer": layer_index,
                        "expert": expert,
                        "selection_count": int(layer.selection_counts[expert].item()),
                        "selection_fraction": selection_distribution[expert],
                        "probability_mass": float(layer.probability_mass[expert].item()),
                        "probability_mass_fraction": probability_distribution[expert],
                        "document_count": int(layer.document_presence[expert].item()),
                        "document_frequency": (
                            float(layer.document_presence[expert].item()) / accumulator.documents
                            if accumulator.documents
                            else float("nan")
                        ),
                    }
                )

    comparisons: dict[str, list[dict[str, Any]]] = {}
    for language, accumulator in language_accumulators.items():
        if language == anchor_language:
            continue
        comparisons[language] = []
        matched_anchor = matched_anchor_accumulators[language]
        for layer_index, (left, right) in enumerate(
            zip(matched_anchor.layers, accumulator.layers)
        ):
            comparison = {
                "layer": layer_index,
                "matched_documents": accumulator.documents,
                "selection_jsd": normalized_jsd(
                    left.selection_counts, right.selection_counts
                ),
                "probability_mass_jsd": normalized_jsd(
                    left.probability_mass, right.probability_mass
                ),
                "shared_gate_mean_delta": (
                    float(right.gate.summary()["mean"]) - float(left.gate.summary()["mean"])
                ),
            }
            comparisons[language].append(comparison)
            layer_rows.append(
                {
                    "row_type": "comparison",
                    "anchor_language": anchor_language,
                    "target_language": language,
                    **comparison,
                }
            )

    report = {
        "schema_version": 1,
        "model": dict(model_metadata),
        "arguments": dict(arguments),
        "metric_semantics": {
            "selection": "Top-k expert assignments; each selected expert counts once.",
            "probability_mass": "Full router-softmax probability mass before top-k selection.",
            "effective_experts": "exp(entropy(p(expert | language))).",
            "top_16_document_persistence": (
                "Mean fraction of documents whose top-16 set contains each of the "
                "corpus-level top-16 experts."
            ),
            "comparisons": (
                "Token-weighted distributions using only IDs matched for that language pair."
            ),
            "shared_gate": "Sigmoid activation of the shared-expert gate.",
        },
        "data": {
            "anchor_language": anchor_language,
            "language_files": {
                key: str(value) for key, value in corpus.language_files.items()
            },
            "anchor_records_seen": corpus.anchor_records_seen,
            "selected_anchor_records": len(anchor_records),
            "missing_from_language": {
                key: len(value) for key, value in corpus.missing_from_language.items()
            },
            "partition_mismatches": corpus.partition_mismatches,
            "languages": {
                language: {
                    "documents": aggregate.documents,
                    "tokens": aggregate.tokens,
                    "windows": aggregate.windows,
                }
                for language, aggregate in language_accumulators.items()
            },
        },
        "languages": language_layers,
        "comparisons": comparisons,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "layer_metrics.csv", [_json_safe(row) for row in layer_rows])
    _write_csv(output_dir / "expert_metrics.csv", [_json_safe(row) for row in expert_rows])
    _write_csv(output_dir / "pair_metrics.csv", [_json_safe(row) for row in pair_rows])
    if make_plots:
        _make_plots(output_dir, language_layers, comparisons, expert_rows, anchor_language)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model ID or local checkpoint path")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer ID/path; defaults to --model")
    parser.add_argument(
        "--language",
        action="append",
        required=True,
        metavar="CODE=PATH",
        help="Language code and its JSONL file; repeat for every language",
    )
    parser.add_argument("--anchor-language", default="eng_Latn")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--id-field", default="warc_record_id")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--partition-field", default="partition")
    parser.add_argument("--max-pairs", type=int, default=1000, help="0 means all anchor records")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_length <= 0:
        parser.error("--max-length must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    try:
        language_files = parse_language_specs(args.language)
        corpus = load_parallel_corpus(
            language_files,
            args.anchor_language,
            max_pairs=args.max_pairs,
            seed=args.seed,
            id_field=args.id_field,
            text_field=args.text_field,
            partition_field=args.partition_field,
        )
        runner, metadata = load_qwen35_runner(
            args.model,
            tokenizer_name=args.tokenizer,
            revision=args.revision,
            local_files_only=args.local_files_only,
            dtype_name=args.dtype,
            device_map=args.device_map,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        try:
            analyze_parallel_corpus(
                runner,
                corpus,
                model_metadata=metadata,
                output_dir=args.output_dir,
                arguments={
                    "max_pairs": args.max_pairs,
                    "max_length": args.max_length,
                    "batch_size": args.batch_size,
                    "seed": args.seed,
                    "id_field": args.id_field,
                    "text_field": args.text_field,
                    "partition_field": args.partition_field,
                    "plots": args.plots,
                },
                make_plots=args.plots,
            )
        finally:
            runner.close()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
