#!/usr/bin/env python3
"""Strip the job-record keys the 32B pool's older monitor rejects (concurrency_group / concurrency_qos / max_active,
null / false in our records; docs/recall_surgery_ambitious.md §14.2) from the given *.job.json files, in place."""
import json
import sys

KEYS = ("concurrency_group", "concurrency_qos", "max_active")


def strip(o):
    if isinstance(o, dict):
        for k in KEYS:
            if k in o:
                assert o[k] in (None, False), (k, o[k])
                del o[k]
        for v in o.values():
            strip(v)


for path in sys.argv[1:]:
    d = json.load(open(path))
    strip(d)
    tmp = path + ".tmp"
    json.dump(d, open(tmp, "w"), indent=1)
    __import__("os").replace(tmp, path)
    print("stripped", path)
