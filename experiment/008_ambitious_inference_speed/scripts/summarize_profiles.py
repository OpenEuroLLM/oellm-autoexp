"""Summarize main-thread wall-time samples after excluding initial prefill."""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'results' / 'batching'


def component(names):
    if any('OellmEngram.' in n for n in names):
        return 'Engram'
    if any('_segmented_arange' in n for n in names):
        return 'FLA sequence indices'
    for name in ['OellmCKDA', 'OellmMHC', 'OellmMoE']:
        if any(name + '.' in n for n in names):
            return name.removeprefix('Oellm')
    return 'other'


def summarize():
    results = []
    for layout in ['ep4', 'tp4']:
        directory = ROOT / layout / 's256_t32768'
        for batch in [32, 256]:
            counts, leaves = collections.Counter(), collections.Counter()
            sources, total = [], 0.0
            for path in sorted(directory.glob(f'profile_b{batch}_rank*_pid*.json')):
                data = json.loads(path.read_text())
                frames = data['shared']['frames']
                if not any('Oellm' in f['name'] for f in frames):
                    continue  # frontend profiles contain waiting/output processing
                main = next(p for p in data['profiles'] if 'MainThread' in p['name'])
                elapsed = 0.0
                for stack, weight in zip(main['samples'], main['weights'], strict=True):
                    elapsed += weight
                    if elapsed < 5.0:
                        continue
                    total += weight
                    counts[component([frames[i]['name'] for i in stack])] += weight
                    if stack:
                        leaf = frames[stack[-1]]
                        leaves[(leaf['name'], leaf.get('file'), leaf.get('line'))] += weight
                sources.append(path.name)
            if not total:
                continue
            results.append({'layout': layout, 'batch': batch,
                'excluded_initial_seconds_per_worker': 5.0,
                'sampled_worker_main_thread_seconds': total,
                'component_percent': {k: v / total * 100 for k, v in counts.most_common()},
                'hot_leaves': [{'function': k[0], 'file': k[1], 'line': k[2],
                               'percent': v / total * 100}
                              for k, v in leaves.most_common(12)], 'sources': sources})
    (ROOT / 'profiles_summary.json').write_text(json.dumps(results, indent=2) + '\n')
    for row in results:
        print(row['layout'], row['batch'],
              {k: round(v, 1) for k, v in row['component_percent'].items()})


if __name__ == '__main__':
    summarize()
