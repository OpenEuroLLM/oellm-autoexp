"""lm_eval entry point with the two shims the MN5 image needs (goal 2026-11-11).

1. ``transformers.data.metrics.squad_metrics`` was removed from recent transformers, and lm-eval's CoQA task
   imports it (``lm_eval/tasks/coqa/utils.py``). The shim registers the module with the standard SQuAD
   normalisation and F1 / exact-match, i.e. exactly what lm-eval's CoQA task scores with.
2. ``sys.argv`` passes straight through to ``lm_eval.__main__.cli_evaluate``, so every lm-eval flag works.

    python scripts/korbi/production/lm_eval_shim.py --model hf --model_args pretrained=<dir>,... --tasks coqa
"""

from __future__ import annotations

import collections
import re
import string
import sys
import types


def _normalize(text: str) -> str:
    """SQuAD's official normalisation (lower, strip punctuation/articles, collapse whitespace)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def compute_exact_match(prediction: str, ground_truth: str) -> int:
    return int(_normalize(prediction) == _normalize(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens, gold_tokens = _normalize(prediction).split(), _normalize(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision, recall = n_same / len(pred_tokens), n_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _install_shim() -> None:
    metrics = types.ModuleType("transformers.data.metrics")
    squad_metrics = types.ModuleType("transformers.data.metrics.squad_metrics")
    squad_metrics.compute_f1 = compute_f1
    squad_metrics.compute_exact_match = compute_exact_match
    squad_metrics.normalize_answer = _normalize
    metrics.squad_metrics = squad_metrics
    sys.modules.setdefault("transformers.data.metrics", metrics)
    sys.modules.setdefault("transformers.data.metrics.squad_metrics", squad_metrics)


if __name__ == "__main__":
    _install_shim()
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()
