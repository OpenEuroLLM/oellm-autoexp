"""Collect operational phase timings and validation parity for checkpoint 3700."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
a = p.parse_args()
h = Path('/gpfs/scratch/ehpc533/hybrid_nucleus')
r = Path('/gpfs/projects/ehpc533/users/korbip/hybrid_v019/autoexp_hf_fast007/experiment/007_fast_hf_export/results')
name = 'ambitious_v4_2ph_fp8_3700'
cases = {
    'baseline': (h/'hf_exports'/name, h/'hf_exports/logs/hf_export_ambitious_v4_3700_fp8/slurm-mn5-pool-p132.0233.log'),
    'candidate': (h/'hf_exports_fast007'/name, h/'kernel_tests/fast_hf_export007b/slurm-mn5-pool-p132.0235.log'),
}
result = {}
for label, (out, log) in cases.items():
    text = log.read_text()
    launch = re.search(r'\[pool\] (\S+) run (\S+) starting .*nodes (\S+)', text)
    start = datetime.fromisoformat(launch[1]).replace(tzinfo=ZoneInfo('Europe/Berlin')).timestamp()
    paths = [out/'.export_done', out/'.check_done', Path(str(out)+'.score.json'), Path(str(out)+'.paired.json')]
    stamps = [start] + [path.stat().st_mtime for path in paths]
    result[label] = {
        'run': launch[2], 'node': launch[3], 'pool_start': launch[1],
        'phase_seconds': dict(zip(['export_including_launch', 'shape_checks', 'score', 'paired'], [y-x for x,y in zip(stamps, stamps[1:])])),
        'ready_seconds_from_pool_start': stamps[-1]-start,
        'score': json.loads(paths[2].read_text()),
        'paired': json.loads(paths[3].read_text()),
    }
base, cand = result['baseline'], result['candidate']
result['speedup'] = {phase: base['phase_seconds'][phase]/cand['phase_seconds'][phase] for phase in base['phase_seconds']}
result['speedup']['end_to_end'] = base['ready_seconds_from_pool_start']/cand['ready_seconds_from_pool_start']
result['parity'] = {}
for kind, ignored in [('score', {'model'}), ('paired', {'model','prefix'})]:
    x = {k:v for k,v in base[kind].items() if k not in ignored}
    y = {k:v for k,v in cand[kind].items() if k not in ignored}
    result['parity'][kind+'_exact'] = x == y
result['parity']['mean_nll_delta'] = cand['score']['mean_nll'] - base['score']['mean_nll']
result['parity']['paired_lm_loss_delta'] = cand['paired']['lm_loss'] - base['paired']['lm_loss']
result['values'] = json.loads((r/'value_comparison.json').read_text())
result['candidate_pipeline_timer'] = json.loads((r/'pipeline_time.json').read_text())
result['reader_benchmark'] = json.loads((r/'reader_benchmark.json').read_text())
result['limitations'] = [
    'Full runs use equivalent different MN5 nodes in one allocation, without cache flushing; shared GPFS load may vary.',
    'Phase times use identical completion markers. Operational totals start at pool launch, including container startup; candidate also repeats two small unit tests, but skips the earlier AB/BA benchmark.',
    'Candidate subprocess pipeline timer separately excludes unit tests, reader benchmark and full-byte comparison.',
    'Controlled reader AB/BA trials use the same node and input, without cache flushing; initial metadata read is recorded separately.',
]
Path(a.out).write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k in {'speedup','parity','values'}}, indent=2))
