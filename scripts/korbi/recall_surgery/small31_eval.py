#!/usr/bin/env python3
"""Phase-1 readout of the small-scale testbed (docs/recall_surgery_ambitious.md §14) on ONE GPU: lm-eval rows with the
HF backend (oellm_ambitious export, trust_remote_code), or the validation loss of the export on fixed v8ml-valid
documents. Results in the layout of the MareNostrum eval launcher, so scripts/korbi/recall_surgery/rescore_fda_squad.py
and small31_table.py read them:

    <out>/<export>/results/<task>_<n_shot>/<model>/results_*.json, samples_<task>_*.jsonl   (samples always logged)
    <out>/<export>/done/<task>_<n_shot>.ok
    <out>/<export>/valid_loss.json                                                          (--valid-loss)

    small31_eval.py --export <hf dir> --out <root> --rows coqa:0 winogrande:0 squadv2:10:4000   # task:shots[:limit]
    small31_eval.py --export <hf dir> --out <root> --rows ruler:4096          # the RULER subset at one length
    small31_eval.py --export <hf dir> --out <root> --valid-loss --valid-root <v8ml-valid dir>

Rows already done (marker) are skipped. A failing row is logged and the next one runs; exit status 1 at the end.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

# the RULER subset (synthetic retrieval / tracking / QA at a fixed context length; cwe / fwe / hotpot left out)
RULER_TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multivalue",
               "niah_multiquery", "ruler_vt", "ruler_qa_squad"]
VALID_SETS = ["all", "web_en", "multilingual", "code", "math", "reasoning"]


def task_objects(names: list[str]) -> list:
    """lm-eval task objects; old-style task classes whose doc_to_text / doc_to_target take only (doc) (SQuAD2) get
    the extra fewshot argument of lm-eval's core dropped (as ladder17b_evaluate._one_arg_compat). The patched objects
    themselves go to simple_evaluate (a fresh load re-imports the class)."""
    import inspect

    from lm_eval.tasks import TaskManager

    loaded = TaskManager().load_task_or_group(names)
    objs = []
    for name in names:
        obj = loaded[name]
        cls = type(obj)
        for meth in ("doc_to_text", "doc_to_target"):
            fn = getattr(cls, meth, None)
            if fn is None or getattr(fn, "_small31_compat", False):
                continue
            params = [q for q in inspect.signature(fn).parameters.values()
                      if q.kind in (q.POSITIONAL_ONLY, q.POSITIONAL_OR_KEYWORD)]
            if len(params) != 2:  # (self, doc) only
                continue

            def wrapped(self, doc, *_args, _fn=fn, **_kw):
                return _fn(self, doc)

            wrapped._small31_compat = True
            setattr(cls, meth, wrapped)
        objs.append(obj)
    return objs


def run_rows(a) -> int:
    import lm_eval
    from lm_eval.loggers import EvaluationTracker

    from lm_eval.models.huggingface import HFLM

    root = Path(a.out) / Path(a.export).name
    done = root / "done"
    done.mkdir(parents=True, exist_ok=True)
    todo = [spec for spec in a.rows if not (done / f"{spec.split(':')[0]}_{spec.split(':')[1]}.ok").exists()]
    if not todo:
        print(f"[small31_eval] all rows done: {a.rows}", flush=True)
        return 0
    # one model for all rows of this process
    lm = HFLM(pretrained=a.export, tokenizer=a.export, trust_remote_code=True, dtype="bfloat16",
              max_length=a.max_length, batch_size=a.batch_size)
    # the production tokenizer's <pad> is id 262144, one past the 262144-row embedding: batched (padded) requests
    # index out of range. Pad with <eos> (padded positions are masked).
    vocab = lm.model.config.vocab_size
    if lm.tokenizer.pad_token_id is None or lm.tokenizer.pad_token_id >= vocab:
        print(f"[small31_eval] pad id {lm.tokenizer.pad_token_id} >= vocab {vocab}: padding with eos "
              f"({lm.tokenizer.eos_token_id})", flush=True)
        lm.tokenizer.pad_token = lm.tokenizer.eos_token
    failed = []
    for spec in a.rows:
        task, n, *lim = spec.split(":")
        limit = float(lim[0]) if lim else a.limit
        key = f"{task}_{n}"
        if (done / f"{key}.ok").exists():
            print(f"[small31_eval] {key}: done", flush=True)
            continue
        t0 = time.time()
        try:
            if task == "ruler":
                tasks, shots, meta = RULER_TASKS, 0, {"max_seq_lengths": [int(n)], "pretrained": a.export}
            else:
                tasks, shots, meta = task_objects([task]), int(n), None
            tracker = EvaluationTracker(output_path=str(root / "results" / key))
            res = lm_eval.simple_evaluate(
                model=lm, tasks=tasks, num_fewshot=shots, log_samples=True, evaluation_tracker=tracker,
                metadata=meta, limit=limit, confirm_run_unsafe_code=True)
            samples = res.pop("samples")
            tracker.save_results_aggregated(results=res, samples=samples)
            for t in samples:
                tracker.save_results_samples(task_name=t, samples=samples[t])
            (done / f"{key}.ok").write_text(json.dumps(res["results"], default=str) + "\n")
            print(f"[small31_eval] {key}: {json.dumps(res['results'], default=str)[:400]} ({time.time() - t0:.0f} s)",
                  flush=True)
        except Exception:
            traceback.print_exc()
            failed.append(key)
    if failed:
        print(f"[small31_eval] FAILED rows: {failed}", flush=True)
    return 1 if failed else 0


def valid_loss(a) -> int:
    """Mean next-token loss of the export on fixed documents of each v8ml-valid set: the first ``--valid-docs``
    documents with >= 256 tokens, each truncated to ``--valid-len`` and prefixed with BOS, one per forward (no
    packing, no padding): the same tokens for every arm and checkpoint."""
    import torch
    import torch.nn.functional as F
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    from transformers import AutoConfig, AutoModelForCausalLM

    root = Path(a.out) / Path(a.export).name
    out = root / "valid_loss.json"
    if out.exists():
        print(f"[small31_eval] valid loss: done ({out})", flush=True)
        return 0
    cfg = AutoConfig.from_pretrained(a.export, trust_remote_code=True)
    bos = cfg.bos_token_id if cfg.bos_token_id is not None else 1
    model = AutoModelForCausalLM.from_pretrained(a.export, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
    result = {}
    for name in VALID_SETS:
        ds = IndexedDataset(os.path.join(a.valid_root, f"{name}_text_document"))
        tot, n_tok, n_doc = 0.0, 0, 0
        for i in range(len(ds)):
            if n_doc >= a.valid_docs:
                break
            doc = ds[i]
            if len(doc) < 256:
                continue
            ids = torch.tensor([bos] + [int(x) for x in doc[: a.valid_len - 1]], device="cuda")[None]
            with torch.no_grad():
                logits = model(ids).logits.float()
            loss = F.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="sum")
            tot += float(loss)
            n_tok += ids.shape[1] - 1
            n_doc += 1
        result[name] = {"loss": tot / max(1, n_tok), "tokens": n_tok, "docs": n_doc}
        print(f"[small31_eval] valid {name}: {result[name]}", flush=True)
    good = [v["loss"] for v in result.values() if v["tokens"]]
    result["mean"] = sum(good) / len(good) if good else math.nan
    out.write_text(json.dumps(result, indent=1) + "\n")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", nargs="*", default=[],
                    help="task:n_shot[:limit] ... (ruler:<length>[:limit] for the RULER subset)")
    ap.add_argument("--batch-size", default="16")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--limit", type=float, default=None, help="documents per task (tests only)")
    ap.add_argument("--valid-loss", action="store_true")
    ap.add_argument("--valid-root", default="/e/data1/datasets/playground/mmlaion/oellm_testing/v8ml-valid")
    ap.add_argument("--valid-docs", type=int, default=64)
    ap.add_argument("--valid-len", type=int, default=4096)
    a = ap.parse_args(argv)
    rc = 0
    if a.rows:
        rc |= run_rows(a)
    if a.valid_loss:
        rc |= valid_loss(a)
    return rc


if __name__ == "__main__":
    sys.exit(main())
