#!/usr/bin/env python3
"""Evaluate a preserved original model, retaining paired per-example evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def audit_sample(row, tokenizer, max_length):
    """Record whether left truncation removed an answer-bearing context."""
    for args in row.get('arguments', []):
        if len(args) != 2 or not isinstance(args[1], dict):
            continue
        prompt, kwargs = args
        ids = tokenizer(str(prompt), add_special_tokens=False)['input_ids']
        budget = max_length - int(kwargs.get('max_gen_toks', 128))
        if budget <= 0:
            raise ValueError('Generation budget exhausts model context')
        retained = tokenizer.decode(ids[-budget:], skip_special_tokens=False)
        gold = row.get('doc', {}).get('value', row.get('target', ''))
        if not isinstance(gold, (list, tuple)):
            gold = [gold]
        gold = [str(g).lower() for g in gold if g]
        return {'prompt_tokens': len(ids), 'context_budget': budget,
                'dropped_front_tokens': max(0, len(ids) - budget),
                'gold_substring_in_prompt': any(g in str(prompt).lower() for g in gold),
                'gold_substring_after_truncation': any(g in retained.lower() for g in gold)}
    return None


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--suite', choices=['focus', 'recall'], required=True)
    p.add_argument('--limit', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--condition', default='baseline', choices=[
        'baseline', 'no_reflection', 'no_neg_alpha', 'both',
        'alpha_gain_0p5', 'alpha_gain_2p0', 'window_512', 'window_1024'])
    a = p.parse_args()
    tasks = [('winogrande', 0), ('squadv2', 10), ('coqa', 0)] if a.suite == 'focus' else [
        ('squad_completion', 0), ('swde', 0), ('fda', 0)]
    a.out.mkdir(parents=True, exist_ok=True)
    model_hash = hashlib.sha256((a.model / 'config.json').read_bytes()).hexdigest()
    pending = []
    for task, shots in tasks:
        dest = a.out / f'{task}.json'
        if dest.exists():
            old = json.loads(dest.read_text())['provenance']
            if (old['model_config_sha256'], old['condition'], old['limit']) != (
                    model_hash, a.condition, a.limit):
                raise ValueError(f'Result identity differs: {dest}')
        else:
            pending.append((task, shots))
    if not pending:
        print('All requested task results already exist.', flush=True)
        return
    sys.path[:0] = [str(a.source), str(a.source / 'lm_scaling')]
    from eval_downstream import build_lm
    from metric_compat import install
    install()
    lm = build_lm(str(a.model), str(a.tokenizer), a.batch_size, 4096, 'cuda',
                  context_prefix='none')
    from interventions import Intervention
    import lm_eval
    from lm_eval.tasks import TaskManager
    with Intervention(lm.model, a.condition) as intervention:
        for task, shots in pending:
            dest = a.out / f'{task}.json'
            start = time.time()
            result = lm_eval.simple_evaluate(
                model=lm, tasks=[task], num_fewshot=shots,
                limit=None if task == 'winogrande' or not a.limit else a.limit,
                bootstrap_iters=0, log_samples=True,
                random_seed=2026, numpy_random_seed=2026,
                torch_random_seed=2026, fewshot_random_seed=2026,
                task_manager=TaskManager(include_path=str(a.source / 'lm_scaling/eval_tasks')))
            samples = a.out / f'{task}.samples.jsonl'
            with samples.open('w') as f:
                for name, rows in result.get('samples', {}).items():
                    for row in rows:
                        doc = row.get('doc', {})
                        row['task'] = name
                        row['document_sha256'] = hashlib.sha256(json.dumps(
                            doc, sort_keys=True, default=str).encode()).hexdigest()
                        row['context_audit'] = audit_sample(row, lm.tok, lm.max_length)
                        f.write(json.dumps(row, default=str) + '\n')
            payload = {'results': result['results'], 'configs': result.get('configs'),
                       'provenance': {'model': str(a.model), 'source': str(a.source),
                                      'model_config_sha256': model_hash,
                                      'condition': a.condition, 'limit': a.limit,
                                      'fewshot': shots, 'seed': 2026, 'context': 4096,
                                      'prefix': 'none', 'elapsed_s': time.time() - start,
                                      'intervention': intervention.stats,
                                      'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES')}}
            tmp = dest.with_suffix('.tmp')
            tmp.write_text(json.dumps(payload, indent=2, default=str) + '\n')
            tmp.replace(dest)
            print(json.dumps({'task': task, 'results': result['results']}), flush=True)


if __name__ == '__main__':
    main()
