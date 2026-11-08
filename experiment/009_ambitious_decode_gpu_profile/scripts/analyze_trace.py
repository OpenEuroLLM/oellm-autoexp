"""Aggregate bounded Nsight traces. Durations are unions where overlap matters."""
import argparse
import gzip
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3
import statistics


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def duration(intervals):
    return sum(b-a for a, b in union(intervals))


def analyze(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    strings = dict(db.execute('select id,value from StringIds'))
    nvtx = [dict(r) for r in db.execute('select * from NVTX_EVENTS')]
    for row in nvtx:
        row['name'] = row['text'] or strings.get(row['textId'], '')
    steps = [r for r in nvtx if re.match(r'STEP/.*/r0/decode\d+/', r['name'])]
    assert steps, f'No decode annotations: {path}'
    steps.sort(key=lambda r:r['start'])
    label = steps[0]['name'].split('/')[1]
    by_count = {int(re.search(r'/decode(\d+)/', r['name'])[1]):r for r in steps}
    coverage = [dict(r) for r in db.execute('select deviceId,min(start) as first_kernel_ns,max(end) as last_kernel_ns,count(*) as kernels from CUPTI_ACTIVITY_KIND_KERNEL group by deviceId')]
    assert len(coverage) == 4, coverage
    common_start = max(r['first_kernel_ns'] for r in coverage)
    common_end = min(r['last_kernel_ns'] for r in coverage)
    eligible = [i for i in range(11,19) if i in by_count and i+1 in by_count and common_start < by_count[i]['start'] and by_count[i+1]['start'] < common_end]
    assert len(eligible) >= 2, (path, coverage, sorted(by_count))
    assert eligible == list(range(eligible[0],eligible[-1]+1))
    start, end = by_count[eligible[0]]['start'], by_count[eligible[-1]+1]['start']
    nsteps = len(eligible)
    wall = end-start
    pid_to_device = dict(db.execute('select distinct globalPid,deviceId from CUPTI_ACTIVITY_KIND_KERNEL'))
    apis = []
    for table in ['CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER']:
        if table in tables:
            for r in db.execute(f'select * from {table} where end>? and start<?', (start,end)):
                r = dict(r)
                r['name'] = strings[r['nameId']]
                r['table'] = table
                apis.append(r)
    # Serialized globalPid equals globalTid with the low 24 thread bits cleared.
    api_index = defaultdict(list)
    api_summary = defaultdict(lambda:[0,0])
    for r in apis:
        api_index[(r['globalTid'] & ~((1<<24)-1),r['correlationId'])].append(r)
        v = api_summary[(r['table'],r['name'])]
        v[0] += 1
        v[1] += min(end,r['end'])-max(start,r['start'])
    scopes = defaultdict(list)
    for r in nvtx:
        if r['end'] and r['name'].startswith(('MODULE/', 'METHOD/')):
            scopes[r['globalTid']].append(r)
    scope_index = {}
    for tid, rows in scopes.items():
        rows.sort(key=lambda r:r['start'])
        max_ends = []
        latest = 0
        for r in rows:
            latest = max(latest, r['end'])
            max_ends.append(latest)
        scope_index[tid] = ([r['start'] for r in rows],rows,max_ends)
    def owner(api):
        if not api or api['globalTid'] not in scope_index:
            return 'outside annotated modules'
        starts, rows, max_ends = scope_index[api['globalTid']]
        i = bisect_right(starts,api['start'])-1
        # Nested ranges; bounded reverse scan keeps the deepest containing one.
        while i >= 0 and max_ends[i] > api['start']:
            r = rows[i]
            if r['start'] <= api['start'] < r['end']:
                return r['name'].split('/')[1]
            i -= 1
        return 'outside annotated modules'
    events = defaultdict(list)
    kernels = defaultdict(lambda:[0,0])
    owners = defaultdict(lambda:[0,0])
    compute_owners = defaultdict(lambda:[0,0])
    unmatched = 0
    for r in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL where end>? and start<?',(start,end)):
        r = dict(r)
        name = strings[r['demangledName']]
        short = strings[r['shortName']]
        a,b = max(start,r['start']),min(end,r['end'])
        candidates = api_index.get((r['globalPid'],r['correlationId']),[])
        api = next((a for a in candidates if a['start']<=r['start']),None)
        if api is None:
            unmatched += 1
        kind = 'collective' if any(x in name.lower() for x in ['nccl','allreduce','all_reduce','reduce_scatter','allgather','all_gather','cross_device_reduce']) else 'compute'
        events[r['deviceId']].append({'start':a,'end':b,'kind':kind,'name':short,
                                     'owner':owner(api),'launch_start':api['start'] if api else None})
        v = kernels[(r['deviceId'],name)]; v[0]+=1; v[1]+=b-a
        v = owners[(r['deviceId'],owner(api))]; v[0]+=1; v[1]+=b-a
        if kind == 'compute':
            v = compute_owners[(r['deviceId'],owner(api))]; v[0]+=1; v[1]+=b-a
    copy_details = defaultdict(lambda:[0,0,0])
    for table in ['CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET']:
        if table not in tables:
            continue
        for r in db.execute(f'select * from {table} where end>? and start<?',(start,end)):
            r = dict(r)
            a,b = max(start,r['start']),min(end,r['end'])
            events[r['deviceId']].append({'start':a,'end':b,'kind':'copy','name':table,'owner':'copy','launch_start':None})
            v=copy_details[(r['deviceId'],str(r.get('copyKind','memset')))]; v[0]+=1; v[1]+=b-a; v[2]+=r.get('bytes',0)
    devices = []
    for device, rows in sorted(events.items()):
        intervals = union([(r['start'],r['end']) for r in rows])
        busy = duration(intervals)
        category = {k:duration([(r['start'],r['end']) for r in rows if r['kind']==k]) for k in ['compute','collective','copy']}
        # Exclusive accounting priority compute > collective > copy; separate unions retained.
        compute = category['compute']
        compute_collective = duration([(r['start'],r['end']) for r in rows if r['kind']!='copy'])
        launch_wait = 0
        prev = start
        launch_starts = defaultdict(list)
        for r in rows:
            if r['launch_start'] is not None:
                launch_starts[r['start']].append(r['launch_start'])
        gaps = []
        for a,b in intervals:
            if a>prev:
                launches = launch_starts[a]
                late = max(0,min(a,min(launches))-prev) if launches else 0
                launch_wait += late
                gaps.append({'start_ns':prev-start,'duration_us':(a-prev)/1e3,'before_next_launch_api_us':late/1e3})
            prev=b
        nk = sum(1 for r in rows if r['kind']!='copy')
        devices.append({'device':device,'busy_pct':100*busy/wall,'kernel_count_per_step':nk/nsteps,
            'compute_union_ms_per_step':compute/1e6/nsteps,
            'collective_union_ms_per_step':category['collective']/1e6/nsteps,
            'copy_union_ms_per_step':category['copy']/1e6/nsteps,
            'exclusive_ms_per_step':{'compute':compute/1e6/nsteps,'collective':(compute_collective-compute)/1e6/nsteps,
                'copy':(busy-compute_collective)/1e6/nsteps,'idle':(wall-busy)/1e6/nsteps},
            'idle_before_next_launch_api_ms_per_step':launch_wait/1e6/nsteps,
            'largest_idle_gaps':sorted(gaps,key=lambda r:r['duration_us'],reverse=True)[:15]})
    counts = [d['kernel_count_per_step'] for d in devices]
    assert (max(counts)-min(counts))/max(counts) < 0.02, ('incomplete/asymmetric kernel coverage', path, counts)
    scope_summary = defaultdict(lambda:[0,0])
    for tid, ranges in scopes.items():
        for r in ranges:
            if r['start'] < end and r['end'] > start:
                v = scope_summary[(tid & ~((1<<24)-1), r['name'].split('/')[1])]
                v[0] += 1
                v[1] += min(end,r['end'])-max(start,r['start'])
    result = {'path':str(path),'label':label,'window_ns':[start,end],'decode_steps':nsteps,'decode_counts':eligible,'kernel_timestamp_coverage':coverage,
        'step_ms':wall/1e6/nsteps,'individual_step_ms':[(by_count[i+1]['start']-by_count[i]['start'])/1e6 for i in eligible],
        'devices':devices,'unmatched_kernel_launches':unmatched,
        'kernels':[{'device':d,'name':name,'count_per_step':v[0]/nsteps,'sum_ms_per_step':v[1]/1e6/nsteps} for (d,name),v in sorted(kernels.items(),key=lambda kv:kv[1][1],reverse=True)],
        'module_kernel_sums':[{'device':d,'owner':name,'count_per_step':v[0]/nsteps,'sum_ms_per_step':v[1]/1e6/nsteps} for (d,name),v in sorted(owners.items(),key=lambda kv:kv[1][1],reverse=True)],
        'module_compute_kernel_sums':[{'device':d,'owner':name,'count_per_step':v[0]/nsteps,'sum_ms_per_step':v[1]/1e6/nsteps} for (d,name),v in sorted(compute_owners.items(),key=lambda kv:kv[1][1],reverse=True)],
        'cpu_module_sums':[{'globalPid':pid,'device':pid_to_device.get(pid),'owner':name,'count_per_step':v[0]/nsteps,'sum_ms_per_step':v[1]/1e6/nsteps} for (pid,name),v in sorted(scope_summary.items(),key=lambda kv:kv[1][1],reverse=True)],
        'cuda_apis':[{'table':t,'name':name,'count_per_step':v[0]/nsteps,'sum_ms_per_step':v[1]/1e6/nsteps} for (t,name),v in sorted(api_summary.items(),key=lambda kv:kv[1][1],reverse=True)],
        'copies':[{'device':d,'kind':kind,'count_per_step':v[0]/nsteps,'ms_per_step':v[1]/1e6/nsteps,'bytes_per_step':v[2]/nsteps} for (d,kind),v in copy_details.items()]}
    # Plot the first middle step only; trace offsets retain exact measured positions.
    step_end = by_count[eligible[0]+1]['start']
    result['timeline'] = {d:[{'kind':r['kind'],'start':(max(start,r['start'])-start)/1e6,'end':(min(step_end,r['end'])-start)/1e6} for r in rows if r['start']<step_end and r['end']>start] for d,rows in events.items()}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('traces',nargs='+',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    for path in args.traces:
        result=analyze(path)
        target=args.out/(result['label']+'.json')
        timeline = result.pop('timeline')
        with gzip.open(target.with_suffix('.timeline.json.gz'), 'wt') as f:
            json.dump(timeline, f, separators=(',',':'))
        target.write_text(json.dumps(result,indent=2)+'\n')
        print(result['label'],round(result['step_ms'],2),[(d['device'],round(d['busy_pct'],1)) for d in result['devices']])


if __name__=='__main__':
    main()
