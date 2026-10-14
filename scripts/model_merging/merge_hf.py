#!/usr/bin/env python3
"""Weighted average of N HF safetensors exports of one architecture, streamed
per tensor chunk.

    merge_hf.py --root EXPORT_ROOT --out NAME --weights linear IN1 IN2 ... INn

Inputs are export names under --root (or paths), OLDEST FIRST; the weights refer to that order.
--weights:
    uniform      1/n each (SMA)
    linear       proportional to 1, 2, ..., n (WMA: the newest checkpoint counts most)
    a,b,c,...    explicit (normalized by their sum)

The result is the correctly rounded weighted mean: sum(w_i * x_i) accumulated in float64 with the
UNNORMALIZED weights (1,1,1 or 1,2,..,n: exact for bf16 inputs), divided once by sum(w_i) and
rounded once to the stored dtype (round-to-nearest-even). So it does not depend on input order or
on how the weights are written, exact cancellations give 0 and exact ties go to even. (fp32
accumulation with per-term scaling, acc += f32(w_i) * x_i, got 0.035% of a 32B checkpoint's
elements wrong by 1 ulp or left ~1e-11 residuals; vanosch1's uniform sum-then-scale happens to be
exact. Verified bit-identical to his merge3 exports and to an independent reference on 674M
elements: dump/model_merging/verify_against_merge3.*.) Output keeps the shard layout and
non-weight files of the LAST input.
Written to NAME.partial and renamed when complete; `.convert_done` (the marker the FLAG prepare
stage waits for) and MERGE_INFO.json are written before the rename. Re-running on a finished
merge is a no-op; a leftover NAME.partial (interrupted run) is removed and redone.

MERGE_INFO.json also records `update_scale`: merging checkpoints theta_1..theta_n with weights
c_i equals keeping theta_1 and scaling every update between theta_{i-1} and theta_i by
sum_{k>=i} c_k (WSM, arXiv:2507.17634) -- i.e. the extra LR decay the merge applies on top of
the schedule the run actually trained with.

Based on vanosch1's uniform downstream_eval_32b/merge_hf.py (the merge3_116k_120k exports).
"""

import argparse
import contextlib
import glob
import json
import os
import shutil
import struct
import time

import numpy as np

CHUNK = 32 * 1024 * 1024  # elements per chunk (256 MiB of float64)


def parse_weights(spec, n):
    """The UNNORMALIZED weights of a --weights spec."""
    if spec == "uniform":
        w = [1.0] * n
    elif spec == "linear":
        w = [float(i + 1) for i in range(n)]
    else:
        w = [float(x) for x in spec.split(",")]
        if len(w) != n:
            raise SystemExit(f"error: {len(w)} weights for {n} inputs")
    if min(w) < 0 or sum(w) <= 0:
        raise SystemExit(f"error: weights must be non-negative with a positive sum: {w}")
    return w


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def index_dir(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        h, base = header(f)
        for k, v in h.items():
            out[k] = (f, base, v)
    return out


def to_f64(raw, dtype):
    if dtype == "BF16":
        return (
            (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16)
            .view(np.float32)
            .astype(np.float64)
        )
    if dtype == "F32":
        return np.frombuffer(raw, np.float32).astype(np.float64)
    raise ValueError(f"unsupported dtype {dtype}")


def from_f64(x, dtype):
    """Round float64 to dtype ONCE, nearest-even (no double rounding through
    fp32)."""
    if not np.isfinite(x).all():
        raise ValueError("non-finite value in merge")
    if dtype == "F32":
        return x.astype(np.float32).tobytes()
    # bf16 keeps 7 of float64's 52 mantissa bits: round at bit 45 in float64, after which the value
    # is exact in fp32 and its top 16 bits are the bf16. Below fp32's normal range (|x| < 2^-126,
    # never a trained weight) the bit position is wrong; round those through fp32 instead.
    u = x.view(np.uint64)
    r = (
        (
            (u + np.uint64((1 << 44) - 1) + ((u >> np.uint64(45)) & np.uint64(1)))
            & ~np.uint64((1 << 45) - 1)
        )
        .view(np.float64)
        .astype(np.float32)
    )
    tiny = np.abs(x) < 2.0**-126
    if tiny.any():
        t = x[tiny].astype(np.float32).view(np.uint32)
        r[tiny] = ((t + 0x7FFF + ((t >> 16) & 1)) >> 16 << 16).astype(np.uint32).view(np.float32)
    return (r.view(np.uint32) >> 16).astype(np.uint16).tobytes()


def merge_chunk(raws, weights, dtype):
    """One chunk: raw bytes of each input -> bytes of the correctly rounded weighted mean."""
    acc = np.zeros(len(raws[0]) // (2 if dtype == "BF16" else 4), np.float64)
    for raw, w in zip(raws, weights):
        acc += w * to_f64(raw, dtype)
    acc /= sum(weights)
    return from_f64(acc, dtype)


def merge(out, ins, weights, label):
    idx = [index_dir(d) for d in ins]
    keys = set(idx[-1])
    if not keys:
        raise SystemExit(f"error: no safetensors in {ins[-1]}")
    for d, i in zip(ins, idx):
        if set(i) != keys:
            raise SystemExit(f"error: tensor names differ in {d}")
        for k in keys:
            a, b = i[k][2], idx[-1][k][2]
            if a["dtype"] != b["dtype"] or a["shape"] != b["shape"]:
                raise SystemExit(
                    f"error: {k} differs in {d}: {a['dtype']}{a['shape']} vs {b['dtype']}{b['shape']}"
                )
    os.makedirs(out)
    t0 = time.time()
    for fpath in sorted(glob.glob(os.path.join(ins[-1], "*.safetensors"))):
        h, _ = header(fpath)
        meta = {"__metadata__": {"format": "pt", "merge": label}}
        off = 0
        names = sorted(h, key=lambda k: h[k]["data_offsets"][0])
        for k in names:
            size = h[k]["data_offsets"][1] - h[k]["data_offsets"][0]
            meta[k] = {
                "dtype": h[k]["dtype"],
                "shape": h[k]["shape"],
                "data_offsets": [off, off + size],
            }
            off += size
        hb = json.dumps(meta, separators=(",", ":")).encode()
        hb += b" " * (-len(hb) % 8)
        handles = {}
        with (
            contextlib.ExitStack() as stack,
            open(os.path.join(out, os.path.basename(fpath)), "wb") as fo,
        ):
            fo.write(struct.pack("<Q", len(hb)))
            fo.write(hb)
            for k in names:
                dtype, shape = h[k]["dtype"], h[k]["shape"]
                isz = 2 if dtype == "BF16" else 4
                nel = int(np.prod(shape)) if shape else 1
                for s in range(0, nel, CHUNK):
                    e = min(nel, s + CHUNK)
                    raws = []
                    for i in idx:
                        f, base, v = i[k]
                        if f not in handles:
                            handles[f] = stack.enter_context(open(f, "rb"))
                        fh = handles[f]
                        fh.seek(base + v["data_offsets"][0] + s * isz)
                        raws.append(fh.read((e - s) * isz))
                    fo.write(merge_chunk(raws, weights, dtype))
        print(f"{os.path.basename(fpath)} done {time.time() - t0:.0f}s", flush=True)
    for f in os.listdir(ins[-1]):
        src = os.path.join(ins[-1], f)
        if (
            not f.endswith(".safetensors")
            and f not in ("MERGE_INFO.json", ".convert_done")
            and os.path.isfile(src)
        ):
            shutil.copy2(src, out)
    return time.time() - t0


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--root", default=".", help="directory the input names and --out are relative to"
    )
    p.add_argument("--out", required=True, help="name (or path) of the merged export")
    p.add_argument("--weights", default="uniform", help="uniform | linear | comma-separated list")
    p.add_argument("inputs", nargs="+", help="exports, oldest first")
    a = p.parse_args()

    ins = [os.path.join(a.root, d) for d in a.inputs]
    out = os.path.join(a.root, a.out)
    if len(ins) < 2:
        raise SystemExit("error: need at least 2 inputs")
    for d in ins:
        if not os.path.isfile(os.path.join(d, "config.json")):
            raise SystemExit(f"error: {d} is no complete HF export (no config.json)")
    weights = parse_weights(a.weights, len(ins))
    if os.path.exists(os.path.join(out, ".convert_done")):
        print(f"{out} already merged, skipping")
        return
    if os.path.exists(out):
        raise SystemExit(f"error: {out} exists without .convert_done -- not ours to overwrite")
    partial = out + ".partial"
    if os.path.exists(partial):
        print(f"removing {partial} left over from an interrupted merge")
        shutil.rmtree(partial)

    total = sum(weights)
    print(f"merge -> {out}")
    for d, w in zip(ins, weights):
        print(f"  {w / total:.4f}  {d}")
    label = f"{a.weights} mean of {len(ins)}"
    secs = merge(partial, ins, weights, label)
    info = {
        "inputs": [os.path.abspath(d) for d in ins],
        "weights_spec": a.weights,
        "weights": [w / total for w in weights],
        # update_scale[i]: factor on the updates between inputs i-1 and i (i >= 1); 1 before input 0
        "update_scale": [sum(weights[i:]) / total for i in range(1, len(ins))],
        "method": "sum(w_i x_i) in float64 / sum(w_i), rounded once to the stored dtype (nearest-even)",
    }
    with open(os.path.join(partial, "MERGE_INFO.json"), "w") as fo:
        json.dump(info, fo, indent=1)
    with open(os.path.join(partial, ".convert_done"), "w") as fo:
        fo.write(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    os.rename(partial, out)
    print(f"done in {secs:.0f}s -> {out}")


if __name__ == "__main__":
    main()
