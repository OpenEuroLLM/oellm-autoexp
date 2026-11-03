"""Analyze the same-model comparison, numerical checks and delayed single profile."""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'results'


def samples(mode, task):
    path = next((ROOT / f'lmeval_{mode}').rglob(f'samples_{task}_*.jsonl'))
    return {(row['doc_id'], row['filter']): row
            for row in map(json.loads, path.read_text().splitlines())}


def main():
    folder = ROOT / 'engram_candidate' / 's256_t32768'
    summary = json.loads((folder / 'summary.json').read_text())
    ranks = [json.loads(p.read_text()) for p in sorted(folder.glob('rank*.json'))]
    checks = [check for rank in summary['engram_checks_by_rank']
              for worker in rank for check in worker]
    result = {'component_parity': {'cases': len(checks),
        'all_outputs_exact': all(c['output_equal'] for c in checks),
        'all_states_exact': all(c['conv_state_equal'] and c['id_state_equal'] for c in checks),
        'max_abs_difference': max(c['max_abs_output_difference'] for c in checks)},
        'performance': {}, 'numerics': {}, 'quality': {}, 'single_profile': {}}
    for mode in ['scoring', 'generation']:
        selected = {variant: [t for t in summary['trials']
            if t['variant'] == variant and t['mode'] == mode][-1]
            for variant in ['reference', 'batched']}
        result['performance'][mode] = {
            'reference_seconds': selected['reference']['seconds'],
            'batched_seconds': selected['batched']['seconds'],
            'speedup': selected['reference']['seconds'] / selected['batched']['seconds'],
            'reference_input_tokens_per_second': selected['reference']['input_tokens_per_second'],
            'batched_input_tokens_per_second': selected['batched']['input_tokens_per_second'],
            'reference_output_tokens_per_second': selected['reference']['output_tokens_per_second'],
            'batched_output_tokens_per_second': selected['batched']['output_tokens_per_second']}
        if mode == 'scoring':
            result['numerics']['mean_nll_delta'] = selected['batched']['mean_nll'] - selected['reference']['mean_nll']
    generated = {}
    for variant in ['reference', 'batched']:
        rows = [row for rank in ranks for trial in rank['trials']
                if trial['variant'] == variant and trial['mode'] == 'generation'
                for row in trial['generated']]
        generated[variant] = {r['index']: r['tokens'] for r in rows}
    a, b = generated['reference'], generated['batched']
    assert a.keys() == b.keys() and len(a) == 1024
    pairs = [(x, y) for i in sorted(a) for x, y in zip(a[i], b[i], strict=True)]
    result['numerics'].update(generated_tokens=len(pairs),
        generated_tokens_matched=sum(x == y for x, y in pairs),
        complete_sequences_matched=sum(a[i] == b[i] for i in a),
        first_tokens_matched=sum(a[i][0] == b[i][0] for i in a))
    control = {row['index']: row['tokens']
               for p in (ROOT / 'batching/ep4/s256_t32768').glob('rank*.json')
               for trial in json.loads(p.read_text())['trials']
               if trial['mode'] == 'generation' for row in trial['generated']}
    assert control.keys() == a.keys()
    result['numerics']['separate_reference_mode_control'] = {
        'tokens': sum(len(a[i]) for i in a),
        'tokens_matched': sum(x == y for i in a for x, y in zip(control[i], a[i], strict=True)),
        'first_tokens_matched': sum(control[i][0] == a[i][0] for i in a),
        'complete_sequences_matched': sum(control[i] == a[i] for i in a)}
    result['decode_scaling'] = summary['decode_scaling_by_variant']
    for task in ['hellaswag', 'gsm8k']:
        a, b = samples('fast', task), samples('batched', task)
        assert a.keys() == b.keys()
        result['quality'][task] = {'unique_docs': len({k[0] for k in a}),
            'records': len(a), 'all_prompt_hashes_match': all(a[k]['prompt_hash'] == b[k]['prompt_hash'] for k in a),
            'records_with_same_correctness': sum(all(a[k][m] == b[k][m]
                for m in a[k]['metrics']) for k in a)}
    single = ROOT / 'engram_single_profile' / 's256_t32768'
    rank = json.loads((single / 'rank0.json').read_text())
    memory = rank['memory_after_init'][0]
    for path in sorted(single.glob('single_profile_rank0_pid*.json')):
        data = json.loads(path.read_text())
        frames = data['shared']['frames']
        profile = next(p for p in data['profiles'] if 'MainThread' in p['name'])
        counts, components, total = collections.Counter(), collections.Counter(), 0.0
        for stack, weight in zip(profile['samples'], profile['weights'], strict=True):
            total += weight
            if stack:
                f = frames[stack[-1]]
                counts[(f['name'], f.get('file'), f.get('line'))] += weight
            names = [frames[i]['name'] for i in stack]
            label = 'other'
            for cls in ['OellmEngram', 'OellmMoE', 'OellmCKDA', 'OellmMHC']:
                if any(cls + '.' in n for n in names):
                    label = cls.removeprefix('Oellm')
                    break
            components[label] += weight
        pid = int(path.stem.split('pid')[-1])
        role = 'worker' if pid == memory['pid'] else 'engine_core' if pid == memory['parent_pid'] else 'frontend'
        result['single_profile'][role] = {'pid': pid, 'sampled_main_thread_seconds': total,
            'component_percent': {k: v / total * 100 for k, v in components.most_common()},
            'hot_leaves': [{'function': k[0], 'file': k[1], 'line': k[2], 'percent': v / total * 100}
                           for k, v in counts.most_common(12)]}
    (ROOT / 'engram_candidate' / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['component_parity', 'performance', 'numerics', 'quality']}, indent=2))


if __name__ == '__main__':
    main()
