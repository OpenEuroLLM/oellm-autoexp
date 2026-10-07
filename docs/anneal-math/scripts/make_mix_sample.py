# ruff: noqa: E701, E702  -- kept verbatim as the record of the code that produced data/heldout_nll
# Fixed sample of anneal-mix docs (first <=2048 tokens), grouped per dataset, for HF NLL probing.
import struct
import numpy as np
import re

MIX = "/e/data1/datasets/playground/mmlaion/oellm/datamix_jupiter_multilingual1_dominant3_complete.txt"
t = open(MIX).read().split()
mix = [(float(t[i]), t[i + 1]) for i in range(0, len(t), 2)]


def short(p):
    s = p.split("complete/")[-1].replace("/megatron-lm", "")
    return re.sub(r"/shard_\d+_text_document|_text_document", "", s)


groups = {}
for w, p in mix:
    groups.setdefault(short(p), []).append((w, p))
top = sorted(groups.items(), key=lambda kv: -sum(w for w, _ in kv[1]))[:40]
seqs, names, ws = [], [], []
rng = np.random.default_rng(123)
for g, shards in top:
    w, p = max(shards)
    with open(p + ".idx", "rb") as fh:
        fh.read(17)
        (code,) = struct.unpack("<B", fh.read(1))
        (S,) = struct.unpack("<Q", fh.read(8))
    L = np.fromfile(p + ".idx", np.int32, S, offset=34)
    P = np.fromfile(p + ".idx", np.int64, S, offset=34 + 4 * S)
    dt = np.dtype({4: np.int32, 8: np.uint16}[code])
    b = np.memmap(p + ".bin", dt, "r")
    cand = np.where(L >= 128)[0]
    for i in rng.choice(cand, min(48, len(cand)), replace=False):
        x = np.array(b[P[i] // dt.itemsize : P[i] // dt.itemsize + min(L[i], 2048)], dtype=np.int64)
        x[x >= 262144] = 0
        seqs.append(x)
        names.append(g)
    ws.append((g, sum(w for w, _ in shards)))
np.savez(
    "/e/scratch/e-sta-openeurollm/poeppel1/v1annealC_probe/mix_sample.npz",
    lens=np.array([len(s) for s in seqs]),
    toks=np.concatenate(seqs),
    names=np.array(names),
    groups=np.array([g for g, _ in ws]),
    weights=np.array([w for _, w in ws]),
)
print(len(seqs), "seqs", sum(len(s) for s in seqs), "tokens", len(ws), "groups")
for g, w in ws:
    print(f"{w:.4f} {g}")
