"""Join decode controls with middle-window trace measurements."""
import argparse
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
a=p.parse_args()
rows=[]
for layout in ['ep4','tp4']:
    ranks=[json.loads(f.read_text()) for f in (a.directory/layout).glob('rank*.json')]
    for batch in [1,32,256]:
        timings=[t for r in ranks for c in r['cases'] if c['batch']==batch for t in c['timings']]
        if not timings:
            continue
        intervals=[(t['last_token_ts']-t['first_token_ts'])/127 for t in timings]
        # Request timestamps share the node's monotonic clock, including DP frontends.
        span=max(t['last_token_ts'] for t in timings)-min(t['first_token_ts'] for t in timings)
        row={'layout':layout,'batch':batch,'requests':len(timings),
             'mean_request_itl_ms':statistics.mean(intervals)*1000,
             'decode_window_output_tok_s':len(timings)*127/span}
        trace=a.directory/'analysis'/f'{layout}_b{batch}.json'
        if trace.exists():
            t=json.loads(trace.read_text())
            row.update(trace_step_ms=t['step_ms'],trace_vs_control_ratio=t['step_ms']/row['mean_request_itl_ms'],
                       gpu_busy_pct=[d['busy_pct'] for d in t['devices']],
                       kernels_per_gpu_step=[d['kernel_count_per_step'] for d in t['devices']])
        rows.append(row)
(a.directory/'timing_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
for row in rows:
    print(json.dumps(row))
