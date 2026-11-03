"""Summarize downloaded batching artifacts, including incomplete/failed cases."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'results' / 'batching'


def summarize():
    rows = []
    for layout in ['ep4', 'tp4']:
        status_path = ROOT / layout / 'status.json'
        statuses = json.loads(status_path.read_text()) if status_path.exists() else []
        extra_path = ROOT / layout / 'status_extra.json'
        if extra_path.exists():
            statuses += json.loads(extra_path.read_text())
        status_map = {(s['max_seqs'], s['token_budget']): s for s in statuses}
        for directory in sorted((ROOT / layout).glob('s*_t*')):
            size, budget = map(int, directory.name[1:].split('_t'))
            row = {'layout': layout, 'max_seqs': size, 'token_budget': budget,
                   **status_map.get((size, budget), {'status': 'in_progress'})}
            path = directory / 'summary.json'
            if path.exists():
                data = json.loads(path.read_text())
                row.update(status='ok', startup_seconds=data['startup_seconds'],
                           prompt_sha256=data['prompt_sha256'])
                for mode in ['scoring', 'generation']:
                    trial = [t for t in data['trials'] if t['mode'] == mode][-1]
                    peaks = trial['scheduler_peaks_by_frontend']
                    running = [max((p['running'] for p in frontend.values()), default=0)
                               for frontend in peaks]
                    memory = [m for frontend in trial['worker_memory_by_frontend'] for m in frontend]
                    row[mode] = {k: trial[k] for k in ['seconds', 'requests_per_second',
                        'input_tokens_per_second', 'output_tokens_per_second']}
                    row[mode].update(
                        running_peak_per_frontend=running,
                        summed_running_peaks=sum(running),
                        preemptions=sum(p['preemptions'] for frontend in peaks for p in frontend.values()),
                        max_worker_peak_allocated_gib=max(m['peak_allocated_bytes'] for m in memory) / 2**30,
                        max_worker_peak_reserved_gib=max(m['peak_reserved_bytes'] for m in memory) / 2**30)
                    if mode == 'scoring':
                        row[mode]['mean_nll'] = trial['mean_nll']
            else:
                row['partial_trials_per_rank'] = {
                    p.stem: [(t['mode'], t['trial'], t['seconds'])
                             for t in json.loads(p.read_text())['trials']]
                    for p in directory.glob('rank*.json')}
            sample_path = directory / 'gpu_samples.json'
            if sample_path.exists():
                used = []
                for sample in json.loads(sample_path.read_text()):
                    for gpu in sample.get('gpus', '').splitlines():
                        used.append(float(gpu.split(',')[1]))
                if used:
                    row['sampled_peak_gpu_used_gib'] = max(used) / 1024
            rows.append(row)
    complete = [r for r in rows if r['status'] == 'ok']
    if complete:
        assert len({r['prompt_sha256'] for r in complete}) == 1
    for row in complete:
        control = next((r for r in complete if r['layout'] == row['layout'] and r['max_seqs'] == 32), None)
        if control:
            for mode in ['scoring', 'generation']:
                row[mode]['speedup_vs_32_control'] = control[mode]['seconds'] / row[mode]['seconds']
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    lines = ['| Layout | Seqs | Token budget | Status | Score input tok/s | Generate output tok/s | Peak GPU GiB |',
             '| --- | ---: | ---: | --- | ---: | ---: | ---: |']
    for row in sorted(rows, key=lambda r: (r['layout'], r['token_budget'], r['max_seqs'])):
        scoring = row.get('scoring', {}).get('input_tokens_per_second', 0)
        generation = row.get('generation', {}).get('output_tokens_per_second', 0)
        lines.append(f"| {row['layout']} | {row['max_seqs']} | {row['token_budget']} | {row['status']} | "
                     f"{scoring:.1f} | {generation:.1f} | {row.get('sampled_peak_gpu_used_gib', 0):.1f} |")
    report = '\n'.join(lines) + '\n'
    (ROOT / 'summary.md').write_text(report)
    print(report)


if __name__ == '__main__':
    summarize()
