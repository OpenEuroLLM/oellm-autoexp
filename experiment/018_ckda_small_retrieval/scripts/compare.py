#!/usr/bin/env python3
"""Summarize paired metrics; only bootstrap numeric per-example scores."""
import argparse
import json
import random
import statistics
from pathlib import Path


def paired(left, right, repeats=2000):
    def rows(path):
        result = {}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (row['task'], str(row['doc_id']), row['document_sha256'])
            if key in result:
                raise ValueError(f'Duplicate example: {key}')
            result[key] = row
        return result
    l, r = rows(left), rows(right)
    if l.keys() != r.keys():
        raise ValueError('Different evaluated documents; refusing paired comparison')
    metrics = set.intersection(*[{
        k for k, v in row.items() if isinstance(v, (float, int))
        and k not in {'doc_id', 'idx'} and not k.endswith('seed')
    } for row in [*l.values(), *r.values()]]) if l else set()
    result = {}
    for metric in sorted(metrics & {'acc', 'acc_norm', 'f1', 'exact_match', 'contains'}):
        delta = [float(l[k][metric]) - float(r[k][metric]) for k in l]
        rng = random.Random(2026)
        draws = sorted(statistics.mean(rng.choices(delta, k=len(delta))) for _ in range(repeats))
        result[metric] = {'n': len(delta), 'left_minus_right': statistics.mean(delta),
                          'paired_bootstrap_95': [draws[int(.025 * repeats)], draws[int(.975 * repeats)]]}
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('ckda', type=Path)
    p.add_argument('kda', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    report = {}
    for left in sorted(a.ckda.glob('*.json')):
        right = a.kda / left.name
        if not right.exists():
            continue
        x, y = json.loads(left.read_text()), json.loads(right.read_text())
        px, py = x['provenance'], y['provenance']
        for key in ['limit', 'fewshot', 'seed', 'context', 'prefix']:
            if px[key] != py[key]:
                raise ValueError(f'{left.stem}: protocol mismatch on {key}')
        report[left.stem] = {'ckda': x['results'], 'kda': y['results'],
                             'paired_numeric_scores': paired(
                                 left.with_suffix('.samples.jsonl'), right.with_suffix('.samples.jsonl'))}
    text = json.dumps(report, indent=2) + '\n'
    if a.out:
        a.out.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
