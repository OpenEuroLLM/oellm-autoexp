import json
import glob
import collections

D = "/e/scratch/e-sta-openeurollm/poeppel1/dump/math_sample/out"
order = [
    "v1anneal_108k",
    "v1annealC_108k",
    "v2anneal_108k",
    "v1anneal_120k",
    "v1annealC_120k_l0fix",
    "v1annealTT_120k",
    "v2anneal_120k",
]
print(
    f"{'checkpoint':22s} "
    + " ".join(f"{t:>9s}" for t in ["AIME24", "AIME25", "AMC23", "MATH500"])
    + "   (share of unforced samples that open with <think>)"
)
for ck in order:
    c = collections.defaultdict(lambda: [0, 0])
    for f in glob.glob(f"{D}/{ck}.shard*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            for s in r["samples"]:
                c[r["task"]][0] += s["text"].lstrip()[:7] == "<think>"
                c[r["task"]][1] += 1
    print(
        f"{ck:22s} "
        + " ".join(f"{c[t][0] / c[t][1]:9.2f}" for t in ["AIME24", "AIME25", "AMC23", "MATH500"])
    )
