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


def _scale_attention_logits(model, x: float) -> dict:
    """Multiply the softmax scale of every fla Attention layer by `x` (DIAGNOSTIC: attention temperature).

    fla's Attention (fla/layers/attn.py) calls the module-level flash_attn_func / flash_attn_varlen_func without a
    softmax_scale (flash-attn's default 1/sqrt(head_dim)) or, without flash-attn, the module-level sdpa_causal
    (torch SDPA, default scale). All three are resolved from the module's globals at call time, so they are wrapped
    there: flash-attn gets softmax_scale = x / sqrt(head_dim), the SDPA fallback gets q * x (the same logits). Every
    Attention module in the model (hybrid layers and recall branches) gets a forward hook that FAILS unless its
    forward went through a wrapped kernel; `state['verified']` collects the modules that did."""
    import fla.layers.attn as fattn
    from fla.layers.attn import Attention

    mods = [(n, m) for n, m in model.named_modules() if isinstance(m, Attention)]
    if not mods:
        raise SystemExit('--attn-logit-scale on a model with no fla Attention layers: nothing to scale')
    state = {'calls': 0, 'verified': set(), 'names': [n for n, _ in mods]}

    def _flash(fn):
        def wrapped(q, k, v, *args, **kw):
            if kw.get('softmax_scale') is not None or args:
                raise SystemExit(f'{fn.__name__}: unexpected softmax_scale / positional args, cannot rescale')
            state['calls'] += 1
            return fn(q, k, v, softmax_scale=x * q.shape[-1] ** -0.5, **kw)
        wrapped._ladder17b_scaled = True
        return wrapped

    patched = []
    for name in ('flash_attn_func', 'flash_attn_varlen_func'):
        fn = getattr(fattn, name, None)
        if fn is not None and not getattr(fn, '_ladder17b_scaled', False):
            setattr(fattn, name, _flash(fn))
            patched.append(name)
    sdpa = fattn.sdpa_causal
    if not getattr(sdpa, '_ladder17b_scaled', False):
        def sdpa_scaled(q, k, v, window_size):
            state['calls'] += 1
            return sdpa(q * x, k, v, window_size)
        sdpa_scaled._ladder17b_scaled = True
        fattn.sdpa_causal = sdpa_scaled
        patched.append('sdpa_causal')

    def pre(mod, _args, _kw=None):
        mod._ladder17b_calls0 = state['calls']

    def post(mod, _args, _out, _name=None):
        if state['calls'] <= mod._ladder17b_calls0:
            raise SystemExit(f'--attn-logit-scale: attention module {_name} ran without a scaled kernel call')
        state['verified'].add(_name)

    for n, m in mods:
        m.register_forward_pre_hook(pre)
        m.register_forward_hook(lambda mod, a, o, _n=n: post(mod, a, o, _name=_n))
    print(f'  ATTENTION LOGIT SCALE x{x} on {len(mods)} fla Attention modules {state["names"]} '
          f'(wrapped {patched}) -- DIAGNOSTIC ONLY, this is not the model that trained', flush=True)
    return state


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
    p.add_argument('--hf-auto', action='store_true',
                   help='a stock HF model (reference, e.g. Qwen3-1.7B-Base): AutoModelForCausalLM instead of the '
                        'ladder exporter\'s explicit classes; same lm-eval wrapper, tasks, shots and documents')
    p.add_argument('--ablate-attn', action='store_true',
                   help='DIAGNOSTIC: zero every fla Attention sublayer (build_lm(ablate_attn=True), the original '
                        'codebase\'s eval_downstream.ablate_attention; the block\'s residual + MLP stay)')
    p.add_argument('--attn-logit-scale', type=float, default=None,
                   help='DIAGNOSTIC: multiply every fla Attention layer\'s softmax scale by this factor')
    a = p.parse_args()
    i, n = (int(x) for x in a.shard.split('/'))
    a.out.mkdir(parents=True, exist_ok=True)
    tag = a.task if n == 1 else f'{a.task}.shard{i}of{n}'
    dest = a.out / f'{tag}.json'
    if dest.exists():
        print(f'{dest} exists', flush=True)
        return
    sys.path[:0] = [str(a.source), str(a.source / 'lm_scaling'), str(Path(__file__).resolve().parent / 'compat'),
                    str(Path(__file__).resolve().parent)]
    from eval_downstream import build_lm
    from metric_compat import install
    install()
    import hf_branch  # the recall-surgery intervention arms' branches (export_native.py's recall_branches)
    hf_branch.install()
    if a.hf_auto:
        import export_hf
        import torch

        def _auto(path, dtype='bfloat16'):
            from transformers import AutoModelForCausalLM
            m = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=getattr(torch, dtype))
            return m.eval(), m.config

        export_hf.load_exported = _auto
    diag = {}
    if a.ablate_attn:
        # ablate_attention leaves the zeroed layers' cache slots empty; only the per-layer, lazily extended
        # FLACache keeps the other layers' indices (LegacyFLACache would append a later layer into the gap)
        from fla.models.utils import Cache, FLACache
        if not issubclass(Cache, FLACache):
            raise SystemExit('--ablate-attn needs the per-layer FLACache (transformers > 4.53.3)')
        diag['ablate_attn'] = True
        lm = build_lm(str(a.model), str(a.tokenizer), a.batch_size, 4096, 'cuda', context_prefix='none',
                      ablate_attn=True)
    else:
        lm = build_lm(str(a.model), str(a.tokenizer), a.batch_size, 4096, 'cuda', context_prefix='none')
    scale_state = None
    if a.attn_logit_scale is not None and a.attn_logit_scale != 1.0:
        if a.ablate_attn:
            raise SystemExit('--attn-logit-scale with --ablate-attn: nothing left to scale')
        scale_state = _scale_attention_logits(lm.model, a.attn_logit_scale)
        diag['attn_logit_scale'] = a.attn_logit_scale
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
    if scale_state is not None:
        missed = sorted(set(scale_state['names']) - scale_state['verified'])
        if missed:
            raise SystemExit(f'--attn-logit-scale: attention modules never ran through a scaled kernel: {missed}')
        diag['attn_scaled_modules'] = len(scale_state['verified'])
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
                              'elapsed_s': round(time.time() - start, 1), **diag}}
    tmp = dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2, default=str) + '\n')
    tmp.replace(dest)
    print(json.dumps({'task': tag, 'results': payload['results'], 'counts': extra}, default=str), flush=True)


if __name__ == '__main__':
    main()
