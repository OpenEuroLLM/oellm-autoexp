#!/usr/bin/env python3
"""The recall-surgery readout of one or more ladder17b_eval output dirs: Winogrande acc, CoQA EM / F1, SQuAD v2
exact / F1 / HasAns F1 / NoAns F1 (shards merged: exact / f1 weighted by documents, HasAns_* by answerable,
NoAns_* by unanswerable documents; best_* thresholds are not mergeable and are left out).

    ladder17b_table.py <out dir> [<out dir> ...]
"""
import json
import sys
from pathlib import Path


def metric(res, *names):
    for n in names:
        for key in (n, f'{n},none', f'{n},remove_whitespace'):
            if key in res and isinstance(res[key], (int, float)):
                return float(res[key])
    return None


def squad(d: Path):
    files = sorted(d.glob('squadv2.shard*of*.json')) or sorted(d.glob('squadv2.json'))
    if not files:
        return None
    tot = {'n': 0, 'has': 0, 'no': 0, 'exact': 0.0, 'f1': 0.0, 'HasAns_f1': 0.0, 'NoAns_f1': 0.0}
    for f in files:
        p = json.loads(f.read_text())
        r, c = p['results'], p.get('counts', {})
        n, has, no = c.get('n_docs', 0), c.get('n_hasans', 0), c.get('n_noans', 0)
        tot['n'] += n; tot['has'] += has; tot['no'] += no
        tot['exact'] += (metric(r, 'exact') or 0) * n
        tot['f1'] += (metric(r, 'f1') or 0) * n
        tot['HasAns_f1'] += (metric(r, 'HasAns_f1') or 0) * has
        tot['NoAns_f1'] += (metric(r, 'NoAns_f1') or 0) * no
    if not tot['n']:
        return None
    return {'n': tot['n'], 'exact': tot['exact'] / tot['n'], 'f1': tot['f1'] / tot['n'],
            'HasAns_f1': tot['HasAns_f1'] / max(tot['has'], 1), 'NoAns_f1': tot['NoAns_f1'] / max(tot['no'], 1),
            'shards': len(files)}


def row(d: Path):
    out = {'dir': str(d)}
    w = d / 'winogrande.json'
    if w.exists():
        r = json.loads(w.read_text())['results']
        out['winogrande'] = metric(r, 'acc')
    c = d / 'coqa.json'
    if c.exists():
        r = json.loads(c.read_text())['results']
        out['coqa_em'], out['coqa_f1'] = metric(r, 'em'), metric(r, 'f1')
    s = squad(d)
    if s:
        out.update({f'squadv2_{k}': v for k, v in s.items()})
    return out


def fmt(v, pct=True):
    if v is None:
        return '-'
    return f'{100 * v:.2f}' if pct and v <= 1.0 else f'{v:.2f}'


def main(argv):
    rows = [row(Path(a)) for a in argv]
    print('| run | Winogrande | CoQA EM | CoQA F1 | SQuAD v2 EM | SQuAD v2 F1 | HasAns F1 | NoAns F1 | SQuAD n |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in rows:
        print(f"| {Path(r['dir']).name} | {fmt(r.get('winogrande'))} | {fmt(r.get('coqa_em'))} | {fmt(r.get('coqa_f1'))} "
              f"| {fmt(r.get('squadv2_exact'), False)} | {fmt(r.get('squadv2_f1'), False)} "
              f"| {fmt(r.get('squadv2_HasAns_f1'), False)} | {fmt(r.get('squadv2_NoAns_f1'), False)} | {r.get('squadv2_n', '-')} |")
    for r in rows:
        (Path(r['dir']) / 'summary.json').write_text(json.dumps(r, indent=2) + '\n')


if __name__ == '__main__':
    main(sys.argv[1:])
