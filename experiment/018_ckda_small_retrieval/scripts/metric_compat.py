"""Provide the removed transformers SQuAD scoring functions used by CoQA."""
import collections
import importlib
import re
import string
import sys
import types


def normalize_answer(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def compute_exact(gold, prediction):
    return int(normalize_answer(gold) == normalize_answer(prediction))


def compute_f1(gold, prediction):
    gold, prediction = normalize_answer(gold).split(), normalize_answer(prediction).split()
    if not gold or not prediction:
        return int(gold == prediction)
    common = sum((collections.Counter(gold) & collections.Counter(prediction)).values())
    if not common:
        return 0.0
    precision, recall = common / len(prediction), common / len(gold)
    return 2 * precision * recall / (precision + recall)


def install():
    name = 'transformers.data.metrics.squad_metrics'
    try:
        importlib.import_module(name)
        return
    except ModuleNotFoundError as exc:
        if not name.startswith(exc.name):
            raise
    data = importlib.import_module('transformers.data')
    metrics = types.ModuleType('transformers.data.metrics')
    metrics.__path__ = []
    squad = types.ModuleType(name)
    squad.compute_f1, squad.compute_exact = compute_f1, compute_exact
    squad.normalize_answer = normalize_answer
    metrics.squad_metrics = squad
    data.metrics = metrics
    sys.modules['transformers.data.metrics'] = metrics
    sys.modules[name] = squad
