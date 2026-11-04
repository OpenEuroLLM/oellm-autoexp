"""Per decode step GPU accounting from an nsys sqlite export (--cuda-graph-trace=node): steps split at
graph launches; per device: step time, kernel-busy union, NCCL union, idle; top kernels by time/step."""
import collections
import json
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
names = dict(db.execute("SELECT id, value FROM StringIds"))
k = db.execute("SELECT start, end, deviceId, shortName, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
rt = db.execute("SELECT start, end, nameId, correlationId, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchall()
launch_corr = {c for s, e, n, c, t in rt if "GraphLaunch" in names.get(n, "")}
by_dev = collections.defaultdict(list)
for s, e, d, n, c in k:
    by_dev[d].append((s, e, names.get(n, str(n)), c))
out = {}
for d, ks in sorted(by_dev.items()):
    # graph-launched kernels share the launch's correlation id: one id per decode step
    steps = collections.OrderedDict()
    for s, e, n, c in ks:
        if c in launch_corr:
            steps.setdefault(c, []).append((s, e, n))
    st = list(steps.values())[2:-2]  # drop capture-boundary steps
    if not st:
        continue
    starts = [x[0][0] for x in st]
    dur = [(b - a) for a, b in zip(starts, starts[1:])]
    busy, nccl, per_name, count = [], [], collections.Counter(), collections.Counter()
    for x in st[:-1]:
        def union(iv):
            tot, cur = 0, None
            for s, e in sorted(iv):
                if cur is None or s > cur[1]:
                    if cur:
                        tot += cur[1] - cur[0]
                    cur = [s, e]
                else:
                    cur[1] = max(cur[1], e)
            return tot + (cur[1] - cur[0] if cur else 0)
        busy.append(union([(s, e) for s, e, n in x]))
        nccl.append(union([(s, e) for s, e, n in x if "nccl" in n.lower()]))
        for s, e, n in x:
            per_name[n] += e - s
            count[n] += 1
    ns = len(st) - 1
    mean = lambda v: sum(v) / len(v) / 1e6  # ms
    out[d] = {"steps": ns, "step_ms": mean(dur), "kernels_per_step": sum(count.values()) / ns,
              "busy_ms": mean(busy), "nccl_ms": mean(nccl),
              "top": [(n[:70], round(t / ns / 1e3, 1), round(count[n] / ns, 1)) for n, t in per_name.most_common(25)]}
print(json.dumps(out, indent=1))
