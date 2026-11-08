#!/usr/bin/env python3
"""Summarize sampled-rank CUDA timelines; comm-only is a proxy, not critical-path proof."""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def union(intervals):
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(hi, out[-1][1])
        else:
            out.append([lo, hi])
    return out


def length(intervals):
    return sum(hi - lo for lo, hi in union(intervals))


def summarize(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as f:
        events = json.load(f)['traceEvents']
    groups = defaultdict(list)
    names = defaultdict(float)
    for e in events:
        if e.get('ph') != 'X' or e.get('cat', '').lower() != 'kernel':
            continue
        lo, dur, name = e['ts'], e['dur'], e.get('name', '')
        kind = 'communication' if any(x in name.lower() for x in ('nccl', 'nvshmem')) else 'compute'
        groups[kind].append((lo, lo + dur))
        names[name] += dur
    communication = length(groups['communication'])
    compute = length(groups['compute'])
    busy = length(groups['communication'] + groups['compute'])
    return dict(path=str(path), units='ms', cuda_busy=busy / 1000,
                communication_union=communication / 1000, compute_union=compute / 1000,
                communication_without_compute=(busy - compute) / 1000,
                caveat='Sampled rank only. Communication without compute includes waits and is not a measured global critical path. Kernel totals overlap.',
                top_kernels=[dict(name=n, summed_ms=t / 1000) for n, t in sorted(names.items(), key=lambda x: -x[1])[:25]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps([summarize(p) for p in args.traces], indent=2))
