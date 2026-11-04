"""Byte-compare every tensor of two HF safetensors exports, independent of the shard layout:
names, dtypes, shapes from the headers, then the raw data bytes in 256 MB pieces on a process pool."""
import argparse
import json
import os
import struct
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PIECE = 256 * 2**20


def tensors(d: Path) -> dict:
    out = {}
    for f in sorted(set(json.loads((d / "model.safetensors.index.json").read_text())["weight_map"].values())):
        with open(d / f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        h.pop("__metadata__", None)
        for k, v in h.items():
            out[k] = (str(d / f), 8 + n + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0], v["dtype"], v["shape"])
    return out


def cmp(task):
    (fa, oa), (fb, ob), n, key = task
    with open(fa, "rb", buffering=0) as x, open(fb, "rb", buffering=0) as y:
        if os.pread(x.fileno(), n, oa) != os.pread(y.fileno(), n, ob):
            return key
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=48)
    a = p.parse_args()
    t0 = time.perf_counter()
    ta, tb = tensors(a.baseline), tensors(a.candidate)
    if ta.keys() != tb.keys():
        raise SystemExit(f"names differ: only baseline {sorted(ta.keys() - tb.keys())[:5]}, only candidate {sorted(tb.keys() - ta.keys())[:5]}")
    tasks = []
    for k in sorted(ta):
        (fa, oa, na, da, sa), (fb, ob, nb, db, sb) = ta[k], tb[k]
        if (na, da, sa) != (nb, db, sb):
            raise SystemExit(f"{k}: {da}{sa} vs {db}{sb}")
        tasks += [((fa, oa + s), (fb, ob + s), min(PIECE, na - s), k) for s in range(0, na, PIECE)]
    with ProcessPoolExecutor(a.workers) as ex:
        bad = sorted({k for k in ex.map(cmp, tasks, chunksize=4) if k})
    files = {}
    for name in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "added_tokens.json", "tokenizer.model", "modeling_oellm_ambitious.py", "configuration_oellm_ambitious.py",
                 "modeling_oellm_hybrid.py", "configuration_oellm_hybrid.py", "mhc_kernels.py"]:
        if (a.baseline / name).exists() or (a.candidate / name).exists():
            files[name] = (a.baseline / name).exists() and (a.candidate / name).exists() and \
                (a.baseline / name).read_bytes() == (a.candidate / name).read_bytes()
    res = {"tensors": len(ta), "tensor_bytes": sum(v[2] for v in ta.values()), "mismatched_tensors": bad,
           "bitwise_equal": not bad, "files_equal": files, "seconds": round(time.perf_counter() - t0, 1)}
    a.out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res), flush=True)
    raise SystemExit(0 if not bad else 1)


if __name__ == "__main__":
    main()
