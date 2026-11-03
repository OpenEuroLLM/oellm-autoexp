"""Compare the complete, fresh v2 pipeline with the frozen sequential baseline."""
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
    'baseline': (h/'hf_exports', h/'hf_exports/logs/hf_export_ambitious_v4_3700_fp8/slurm-mn5-pool-p132.0233.log'),
    'v2': (h/'hf_exports_fast007parallel', h/'kernel_tests/fast_hf_export007parallel/current.log'),
}
result = {}
for label, (root, log) in cases.items():
    text = log.read_text()
    launch = re.search(r'\[pool\] (\S+) run (\S+) starting .*nodes (\S+)', text)
    start = datetime.fromisoformat(launch[1]).replace(tzinfo=ZoneInfo('Europe/Berlin')).timestamp()
    out = root/name
    export_end = (out/'.export_done').stat().st_mtime
    check_end = (out/'.check_done').stat().st_mtime
    score_file, paired_file = Path(str(out)+'.score.json'), Path(str(out)+'.paired.json')
    score_end, paired_end = score_file.stat().st_mtime, paired_file.stat().st_mtime
    result[label] = {
        'run': launch[2], 'node': launch[3], 'pool_start': launch[1],
        'export_including_launch_seconds': export_end-start,
        'shape_check_seconds': check_end-export_end,
        'validation_wall_seconds': max(score_end, paired_end)-check_end,
        'ready_seconds_from_pool_start': max(score_end, paired_end)-start,
        'score': json.loads(score_file.read_text()),
        'paired': json.loads(paired_file.read_text()),
    }
    if label == 'v2':
        for kind in ['score', 'paired']:
            result[label][kind+'_timer'] = json.loads(Path(str(out)+f'.{kind}.time.json').read_text())
        reader = re.search(r'Export reader: metadata_disk_reads=(\d+) metadata_seconds=([\d.]+) read_seconds=([\d.]+) threads=(\d+)', text)
        result[label]['reader'] = dict(zip(['metadata_disk_reads','metadata_seconds','read_seconds','threads'], map(float, reader.groups())))
base, cand = result['baseline'], result['v2']
result['speedup'] = {key: base[key]/cand[key] for key in ['export_including_launch_seconds','validation_wall_seconds','ready_seconds_from_pool_start']}
result['parity'] = {}
for kind, ignored in [('score', {'model'}), ('paired', {'model','prefix'})]:
    x = {k:v for k,v in base[kind].items() if k not in ignored}
    y = {k:v for k,v in cand[kind].items() if k not in ignored}
    result['parity'][kind+'_exact'] = x == y
result['parity']['mean_nll_delta'] = cand['score']['mean_nll']-base['score']['mean_nll']
result['parity']['paired_lm_loss_delta'] = cand['paired']['lm_loss']-base['paired']['lm_loss']
result['parity']['paired_max_sample_loss_sum_delta'] = max(abs(x-y) for x,y in zip(base['paired']['per_sample_loss_sum'],cand['paired']['per_sample_loss_sum'], strict=True))
for kind, field in [('score','nonfinite_docs'),('paired','nonfinite_samples')]:
    assert not base[kind][field] and not cand[kind][field], (kind, field)
result['values'] = json.loads((r/'parallel/value_comparison.json').read_text())
result['pipeline_timer'] = json.loads((r/'parallel/pipeline_time.json').read_text())
result['limitations'] = [
    'Equivalent different MN5 nodes and shared GPFS; no cache flushing. Full-run timing is observational.',
    'Both runs reserve one four-GPU node. v2 runs the same ordinary and paired checks concurrently on separate two-GPU pairs.',
    'Operational total starts at pool launch; pipeline timer excludes capacity preflight and full-byte comparison.',
    'v2 validation changes physical GPU placement, but exported tensors and validation inputs are unchanged.',
]
Path(a.out).write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k in {'speedup','parity','values'}}, indent=2))
