"""Aggregate paired controls and quantify numerical changes against A/A noise."""
import argparse
import json
from pathlib import Path
import statistics
import math

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
ranks=[json.loads(f.read_text()) for f in sorted((a.directory/'paired').glob('rank*.json'))]
assert len(ranks)==4
variants=['reference0','reference1','fused','graphs','graphs_repeat']
summary={'decode':[],'scoring':[],'quality':[]}
initial=[r['trials'][1]['graphs'] for r in ranks if len(r['trials'])>1]
initial_captures=sum(v['captures'] for group in initial for stat in group for v in stat.values())
initial_replays=sum(v['replays'] for group in initial for stat in group for v in stat.values())
for variant in variants:
    for batch in [1,32,256]:
        entries=[t for r in ranks for t in r['trials'] if t['variant']==variant and t['mode']=='decode' and t['batch']==batch]
        if len(entries)!=4:continue
        req=[x for t in entries for x in t['requests']]
        first=min(x['first_token_ts'] for x in req);last=max(x['last_token_ts'] for x in req)
        itl=statistics.mean((x['last_token_ts']-x['first_token_ts'])/127 for x in req)
        stats=[g for e in entries for g in e['graphs']]
        summary['decode'].append({'variant':variant,'batch':batch,'requests':len(req),
            'decode_output_tok_s':len(req)*127/(last-first),'mean_request_itl_ms':itl*1000,
            'wall_seconds':max(t['wall_seconds'] for t in entries),
            'min_gpu_free_gib':min(m['free_bytes'] for t in entries for m in t['memory'])/2**30,
            'peak_gpu_allocated_gib':max(m['peak_allocated_bytes'] for t in entries for m in t['memory'])/2**30,
            'graph_captures':sum(v['captures'] for s in stats for v in s.values())-initial_captures,
            'graph_replays':sum(v['replays'] for s in stats for v in s.values())-initial_replays,
            'graph_fallbacks':sum(v['fallbacks'] for s in stats for v in s.values())})
    records=[t for r in ranks for t in r['trials'] if t['variant']==variant and t['mode']=='scoring']
    if len(records)==4:
        scores=[s for t in records for row in t['scores'] for s in row]
        assert all(math.isfinite(x) for x in scores)
        summary['scoring'].append({'variant':variant,'mean_nll':-statistics.mean(scores),
            'input_tok_s':len(scores)/max(t['seconds'] for t in records),'tokens':len(scores)})
for variant in variants[1:]:
    expected=[];actual=[];token_same=0;total=0
    for r in ranks:
        ref=next((t for t in r['trials'] if t['variant']=='reference0' and t['mode']=='scoring'),None)
        new=next((t for t in r['trials'] if t['variant']==variant and t['mode']=='scoring'),None)
        if ref and new:
            expected.extend(s for row in ref['scores'] for s in row)
            actual.extend(s for row in new['scores'] for s in row)
        for t in r['trials']:
            if t['variant']!=variant or t['mode']!='decode':continue
            control=next(t0 for t0 in r['trials'] if t0['variant']=='reference0' and t0['mode']=='decode' and t0['batch']==t['batch'])
            for x,y in zip(t['requests'],control['requests']):
                token_same+=sum(a==b for a,b in zip(x['tokens'],y['tokens']));total+=len(x['tokens'])
    if expected:
        dif=[a-b for a,b in zip(actual,expected)]
        summary['quality'].append({'variant':variant,'scored_tokens':len(dif),'mean_nll_delta':-statistics.mean(dif),
            'rms_logprob_delta':math.sqrt(statistics.mean(x*x for x in dif)),
            'max_abs_logprob_delta':max(map(abs,dif)),'generated_token_agreement':token_same/total if total else None})
(a.directory/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
for key,rows in summary.items():
 print(key)
 for row in rows:print(json.dumps(row))
