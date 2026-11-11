#!/usr/bin/env python3
"""One lm-eval task (or a shard of it) on an exported native ladder model, through the original codebase's own
lm-eval wrapper (lm_scaling/eval_downstream.build_lm, export_hf.load_exported) and its task definitions
(lm_scaling/eval_tasks), with the CoQA scoring functions transformers removed (018's metric_compat.install).
Derived from experiment/018_ckda_small_retrieval/scripts/evaluate.py; same seeds (2026), context 4096, no prefix.

    ladder17b_evaluate.py --source <code dir with lm_scaling> --model <export> --tokenizer <dir>
        --task squadv2 --shots 10 [--limit N] [--shard i/n] --out <dir>

Writes <out>/<task>[.shard<i>of<n>].json: lm-eval's metrics, the evaluated doc count and, for SQuAD v2, the
HasAns / NoAns document counts (so shards can be merged: ladder17b_table.py).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def _one_arg_compat(task_obj):
    """lm-eval's core (0.4.13 in the image) calls doc_to_text / doc_to_target(doc, <fewshot cfg field>) in
    fewshot_context; old-style task classes (SQuAD2) take only (doc). Patch the task's class so the extra
    argument is dropped (the prompt is the class's own either way). Classes are shared with the instances
    simple_evaluate builds, so patching the class covers them."""
    import inspect
    cls = type(task_obj)
    patched = []
    for name in ('doc_to_text', 'doc_to_target'):
        fn = getattr(cls, name, None)
        if fn is None or getattr(fn, '_ladder17b_compat', False):
            continue
        params = [p for p in inspect.signature(fn).parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if len(params) != 2:  # (self, doc) only
            continue

        def wrapped(self, doc, *_args, _fn=fn, **_kw):
            return _fn(self, doc)
        wrapped._ladder17b_compat = True
        setattr(cls, name, wrapped)
        patched.append(f'{cls.__name__}.{name}')
    return patched


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--shots', type=int, required=True)
    p.add_argument('--limit', type=int, default=0, help='first N documents (0: all)')
    p.add_argument('--shard', default='0/1', help='i/n: every n-th of the (limited) documents, offset i')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    i, n = (int(x) for x in a.shard.split('/'))
    a.out.mkdir(parents=True, exist_ok=True)
    tag = a.task if n == 1 else f'{a.task}.shard{i}of{n}'
    dest = a.out / f'{tag}.json'
    if dest.exists():
        print(f'{dest} exists', flush=True)
        return
    sys.path[:0] = [str(a.source), str(a.source / 'lm_scaling'), str(Path(__file__).resolve().parent / 'compat')]
    from eval_downstream import build_lm
    from metric_compat import install
    install()
    lm = build_lm(str(a.model), str(a.tokenizer), a.batch_size, 4096, 'cuda', context_prefix='none')
    import lm_eval
    from lm_eval.tasks import TaskManager
    tm = TaskManager(include_path=str(a.source / 'lm_scaling/eval_tasks'))
    # the document indices: first `limit`, then every n-th from offset i
    task_obj = tm.load_task_or_group([a.task])[a.task] if hasattr(tm, 'load_task_or_group') else None
    total = None
    if task_obj is not None:
        docs = task_obj.eval_docs
        total = len(docs)
        patched = _one_arg_compat(task_obj)
        if patched:
            print(f'[ladder17b_evaluate] one-argument task API patched: {patched}', flush=True)
    elif a.task.startswith('squad'):
        raise SystemExit('cannot load the task object to apply the SQuAD2 task-API patch')
    upto = a.limit if a.limit else total
    samples = None
    if n > 1 or a.limit:
        if upto is None:
            raise SystemExit('cannot determine the document count for --limit / --shard')
        samples = {a.task: list(range(i, min(upto, total), n))}
    start = time.time()
    # the patched task object itself (lm-eval re-imports a !function task class on every load, so a class patch
    # would not reach a freshly built instance)
    kw = dict(model=lm, tasks=[task_obj] if task_obj is not None else [a.task], num_fewshot=a.shots, bootstrap_iters=0, log_samples=True,
              random_seed=2026, numpy_random_seed=2026, torch_random_seed=2026, fewshot_random_seed=2026,
              task_manager=tm)
    try:
        result = lm_eval.simple_evaluate(samples=samples, **kw)
    except TypeError as exc:  # an lm-eval without per-document selection: first `limit` docs, no sharding
        if 'samples' not in str(exc) or n > 1:
            raise
        result = lm_eval.simple_evaluate(limit=a.limit or None, **kw)
    rows = result.get('samples', {}).get(a.task, [])
    extra = {'n_docs': len(rows)}
    if a.task.startswith('squad'):
        has = sum(1 for r in rows if (r.get('doc', {}).get('answers', {}) or {}).get('text'))
        extra.update(n_hasans=has, n_noans=len(rows) - has)
    with (a.out / f'{tag}.samples.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps({k: r.get(k) for k in ('doc_id', 'filtered_resps', 'target', 'acc', 'f1', 'em',
                                                      'exact', 'contains')}, default=str) + '\n')
    payload = {'results': result['results'].get(a.task, result['results']), 'counts': extra,
               'provenance': {'model': str(a.model), 'source': str(a.source), 'task': a.task, 'shots': a.shots,
                              'limit': a.limit, 'shard': a.shard, 'seed': 2026, 'context': 4096,
                              'model_config_sha256': hashlib.sha256((a.model / 'config.json').read_bytes()).hexdigest(),
                              'intervention': os.getenv('SKDA_INTERVENTION', ''),
                              'elapsed_s': round(time.time() - start, 1)}}
    tmp = dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2, default=str) + '\n')
    tmp.replace(dest)
    print(json.dumps({'task': tag, 'results': payload['results'], 'counts': extra}, default=str), flush=True)


if __name__ == '__main__':
    main()
