"""Summarize downloaded experiment artifacts using only the Python standard library."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'results'


def load(path):
    return json.loads((ROOT / path).read_text())


def differences(reference, candidate):
    assert len(reference) == len(candidate)
    delta = [b - a for a, b in zip(reference, candidate, strict=True)]
    return {'tokens': len(delta), 'mean_nll_delta': -statistics.mean(delta),
            'mean_abs_logprob_difference': statistics.mean(map(abs, delta)),
            'max_abs_logprob_difference': max(map(abs, delta))}


def hf_comparison(reference, candidate):
    x = sum(reference['trials'][1]['prompt_logprobs'], [])
    y = sum(candidate['trials'][1]['prompt_logprobs'], [])
    xg, yg = sum(reference['generated'], []), sum(candidate['generated'], [])
    return {**differences(x, y),
            'warm_seconds': candidate['trials'][1]['seconds'],
            'warm_speedup': reference['trials'][1]['seconds'] / candidate['trials'][1]['seconds'],
            'generation_speedup': reference['generation_seconds_including_prefill'] /
                                  candidate['generation_seconds_including_prefill'],
            'generated_tokens': len(xg),
            'generated_tokens_matched': sum(a == b for a, b in zip(xg, yg, strict=True))}


def vllm_rows(directory, trial=1):
    rows = []
    for file in sorted((ROOT / directory).glob('rank*.json')):
        rows.extend(json.loads(file.read_text())['trials'][trial]['rows'])
    return sorted(rows, key=lambda row: row['index'])


def vllm_comparison(reference, candidate, reference_trial=1, candidate_trial=1):
    a, b = vllm_rows(reference, reference_trial), vllm_rows(candidate, candidate_trial)
    assert [r['prompt_token_ids'] for r in a] == [r['prompt_token_ids'] for r in b]
    x = [p for r in a for p in r['prompt_logprobs'] if p is not None]
    y = [p for r in b for p in r['prompt_logprobs'] if p is not None]
    xg, yg = sum([r['generated'] for r in a], []), sum([r['generated'] for r in b], [])
    return {**differences(x, y), 'generated_tokens': len(xg),
            'generated_tokens_matched': sum(a == b for a, b in zip(xg, yg, strict=True))}


def task_results(directory):
    results = {}
    for file in (ROOT / directory / 'results').rglob('results_*.json'):
        data = json.loads(file.read_text())
        for name, metrics in data['results'].items():
            samples = list(file.parent.glob('samples_*.jsonl'))
            results[name] = {'metrics': metrics,
                'samples': {p.name.split('_')[1]: {
                    'records': len(p.read_text().splitlines()),
                    'unique_doc_ids': len({json.loads(line)['doc_id']
                                          for line in p.read_text().splitlines()}),
                    'filters': sorted({json.loads(line)['filter']
                                       for line in p.read_text().splitlines()})}
                            for p in samples}}
    return results


def task_comparison():
    result = {}
    for task in ['hellaswag', 'gsm8k']:
        samples = []
        for mode in ['reference', 'fast']:
            files = list((ROOT / f'lmeval_{mode}').rglob(f'samples_{task}_*.jsonl'))
            if not files:
                break
            assert len(files) == 1
            rows = [json.loads(line) for line in files[0].read_text().splitlines()]
            samples.append({(row['doc_id'], row['filter']): row for row in rows})
        if len(samples) != 2:
            continue
        a, b = samples
        assert a.keys() == b.keys()
        assert all(a[k]['prompt_hash'] == b[k]['prompt_hash'] for k in a)
        result[task] = {'unique_docs': len({k[0] for k in a}),
            'doc_filter_records': len(a), 'all_prompt_hashes_match': True,
            'records_with_identical_correctness': sum(
                all(a[k][metric] == b[k][metric] for metric in a[k]['metrics']) for k in a)}
    return result


def main():
    result = {'hf': {}, 'vllm': {}, 'numerics': {}, 'tasks': {}}
    reference = load('optimized/hf_reference.json')
    for name, path in [('fast', 'optimized/hf_fast.json'),
                       ('grouped_reference_mhc', 'quality/hf_grouped_only.json'),
                       ('mapped_tables', 'final/hf_mmap.json')]:
        if (ROOT / path).exists():
            result['hf'][name] = hf_comparison(reference, load(path))
    for path in sorted(ROOT.rglob('summary.json')):
        data = json.loads(path.read_text())
        if 'layout' in data:
            result['vllm'][str(path.parent.relative_to(ROOT))] = data
    for name, a, b in [('matched_tp4_reference_to_ep4_fast', 'optimized/tp4_tuned', 'optimized/ep4'),
                       ('matched_tp4_reference_to_tp4_fast', 'optimized/tp4_tuned', 'optimized/tp4')]:
        result['numerics'][name] = vllm_comparison(a, b)
    result['numerics']['ep4_fast_repeat_variation'] = vllm_comparison(
        'optimized/ep4', 'optimized/ep4', 0, 1)
    for name in ['lmeval_fast', 'lmeval_reference']:
        result['tasks'][name] = task_results(name)
    result['task_comparison'] = task_comparison()
    if (ROOT / 'final/hf_mmap.json').exists():
        result['hf']['mapped_vs_fast'] = hf_comparison(
            load('optimized/hf_fast.json'), load('final/hf_mmap.json'))
    (ROOT / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['hf', 'numerics', 'tasks']}, indent=2))


if __name__ == '__main__':
    main()
