"""Split one lm-eval row into document parts that run on different engines, and merge them.

A part row is named ``<half>/<task>/<n_shot>@<k>of<K>``: it evaluates every K-th document,
starting at k, of each leaf task of <task> (``simple_evaluate(samples=...)``), and keeps the
per-document metric values. When all K parts of a row are there, ``merge`` puts the values of
all parts into lm-eval's own accumulators and runs lm-eval's aggregation
(``evaluator_utils._process_results``: the task aggregations, stderrs and group weighting), so
the merged results are those of the unsplit row up to the order of the documents.

Made for the long multiple-choice rows (hellaswag, mmlu, global_mmlu_full_*: per-document
metrics, mean aggregations), which decide the wall clock of a short eval on many nodes.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

_PART = re.compile(r"^(?P<base>.+)@(?P<k>\d+)of(?P<n>\d+)$")


def parse(name: str) -> tuple[str, int | None, int | None]:
    """``vllm/mmlu/5@1of3`` -> (``vllm/mmlu/5``, 1, 3); a whole row -> (name, None, None)."""
    m = _PART.match(name)
    if not m:
        return name, None, None
    k, n = int(m["k"]), int(m["n"])
    if not 0 <= k < n:
        raise ValueError(f"{name}: part {k} of {n}")
    return m["base"], k, n


def part_key(task: str, n_shot: int, k: int, n: int) -> str:
    return f"{task}_{n_shot}@{k}of{n}"


def part_samples(task_manager, task: str, k: int, n: int) -> dict[str, list[int]]:
    """Document indices of part k of n for every leaf task of ``task``."""
    loaded = task_manager.load([task])
    return {
        name: list(range(k, len(obj.eval_docs), n)) for name, obj in sorted(loaded["tasks"].items())
    }


def raw_metrics(samples: dict[str, list[dict]]) -> dict[str, list]:
    """The per-document metric values of an evaluation's logged samples, as lm-eval's
    accumulators hold them: {leaf: [[metric, filter, [values in document order]], ...]}."""
    out: dict[str, list] = {}
    for leaf, examples in samples.items():
        acc: dict[tuple[str, str], list] = defaultdict(list)
        for ex in sorted(examples, key=lambda e: e["doc_id"]):
            for metric in ex["metrics"]:
                acc[(metric, ex["filter"])].append(ex[metric])
        out[leaf] = [[m, f, v] for (m, f), v in acc.items()]
    return out


def write_part(path: Path, task: str, n_shot: int, k: int, n: int, samples: dict[str, list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"task": task, "n_shot": n_shot, "part": [k, n], "raw": raw_metrics(samples)}))
    tmp.replace(path)


def merge(task_manager, task: str, n_shot: int, parts: list[dict], bootstrap_iters: int = 100000) -> dict:
    """lm-eval's results dict of the whole row from the parts' ``raw`` metrics."""
    from lm_eval.evaluator_utils import _process_results

    got = sorted(p["part"][0] for p in parts)
    n = parts[0]["part"][1]
    if got != list(range(n)):
        raise ValueError(f"{task}: parts {got} of {n}")
    loaded = task_manager.load([task])
    for obj in loaded["tasks"].values():
        if obj.get_config("num_fewshot") != 0:  # as simple_evaluate: a 0 in the task config stays
            obj.set_config(key="num_fewshot", value=n_shot)
    acc = {
        name: {"task": obj, "raw_metrics": defaultdict(list), "logged_samples": []}
        for name, obj in loaded["tasks"].items()
    }
    for p in sorted(parts, key=lambda p: p["part"][0]):
        for leaf, entries in p["raw"].items():
            for metric, filt, values in entries:
                acc[leaf]["raw_metrics"][(metric, filt)].extend(values)
    return _process_results(acc, loaded["groups"], bootstrap_iters)._to_eval_results()
