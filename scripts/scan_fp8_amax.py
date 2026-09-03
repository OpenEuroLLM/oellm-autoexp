#!/usr/bin/env python3
"""Read the TE FP8 amax/scale history out of the torch_dist checkpoints.

Extends the fp8 amax evidence backwards past iteration 40,000 -- the offline
audit only ever covered 40k-72k, which starts *after* the 2026-08-28 stack swap
at iteration 34,455, so it cannot say whether the activation growth began with
the swap or was there all along. This reads every checkpoint we still have.

scale = fp8_max / amax, so a FALLING scale means the activations are GROWING.

Needs the NVIDIA driver visible, not a compute node: TE writes _extra_state as a
pickle whose unpickling imports megatron.core -> transformer_engine ->
libcuda.so.1. The JUPITER login nodes DO have a GH200 and do have libcuda, so
`apptainer exec --nv` is enough. `--nv` is what binds the host driver libraries
into the container; omit it and the import fails with
`libcuda.so.1: cannot open shared object file` even on a machine that has a GPU,
which reads exactly like "no GPU here" and is the trap this note exists to
prevent. Nothing in this script calls into CUDA; it only needs the import to
succeed.

  # login node, no allocation
  apptainer exec --nv /e/project1/e-sta-openeurollm/container/\
MegatronTraining-JUPITER-te218-fa3_aarch64_202608280932.sif \
    python3 scripts/scan_fp8_amax.py <checkpoints-dir> --csv docs/64k-debug/data/fp8_amax.csv
"""

import argparse
import csv
import io
import pickle
import re
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

DUMPED = object()  # sentinel: dump mode found and printed an entry

# The four FP8 GEMMs per layer. Their _extra_state holds the delayed-scaling
# bookkeeping; the layernorm ones do not run in FP8 and carry nothing useful.
FP8_LAYERS = [
    "decoder.layers.self_attention.linear_qkv",
    "decoder.layers.self_attention.linear_proj",
    "decoder.layers.mlp.linear_fc1",
    "decoder.layers.mlp.linear_fc2",
]


class _Stub:
    """Stands in for megatron/TE classes referenced by the pickle.

    Nothing we read (scale_*, amax_history_*) is one of them, so they
    never get touched -- stubbing them is what keeps this a CPU/login-
    node job instead of needing a GPU node just to satisfy an import of
    libcuda.so.1.
    """

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})


def _load_cpu(b):
    """Newer checkpoints pickle the FP8 state as CUDA tensors, so the default
    storage loader tries to init CUDA and dies on libcuda.so.1.

    Force CPU.
    """
    return torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(("megatron", "transformer_engine", "nemo", "apex")):
            return _Stub
        if (module, name) == ("torch.storage", "_load_from_bytes"):
            return _load_cpu
        return super().find_class(module, name)


def decode(blob):
    """_extra_state is a uint8 tensor holding a RAW pickle stream (0x80 0x04
    ...), not a torch.save file -- torch.load on it fails with "Invalid magic
    number"."""
    t = blob[0] if isinstance(blob, list) else blob
    if isinstance(t, io.BytesIO):
        t.seek(0)
        t = t.read()
    elif hasattr(t, "numpy"):
        t = t.numpy().tobytes()
    # Order matters. TE 2.18 serialises the extra state through a HELPER
    # FUNCTION, not a plain dict: pickle calls it and uses the return value. The
    # stubbing unpickler below hands back a _Stub class for that symbol, so the
    # call yields a _Stub INSTANCE and the payload decodes to an object that is
    # not a dict -- no exception, just a silently empty scan. Every post-swap
    # checkpoint read that way looked like it had no FP8 state at all.
    #
    # So try the real classes first; that works whenever transformer_engine
    # imports, which on a login node means running under `apptainer exec --nv`.
    # Fall back to the stubbing unpickler only when the real import is
    # unavailable, and accept its result only if it actually produced a dict.
    try:
        return pickle.loads(t)
    except Exception:
        pass
    try:
        obj = _Unpickler(io.BytesIO(t)).load()
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return torch.load(io.BytesIO(t), weights_only=False, map_location="cpu")


def flatten(d, prefix=""):
    """TE has moved these keys around between versions, so walk the whole dict
    and keep every tensor rather than hardcoding a layout."""
    out = {}
    for k, v in d.items():
        name = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, name + "."))
        elif torch.is_tensor(v):
            out[name] = v.float()
    return out


def scan(ckpt, dump=False):
    reader = dcp.FileSystemReader(ckpt)
    meta = reader.read_metadata().state_dict_metadata

    keys = [
        k
        for k in meta
        if "_extra_state" in k and any(k.startswith(p + "._extra_state") for p in FP8_LAYERS)
    ]
    if dump:
        allx = [k for k in meta if "_extra_state" in k]
        print(f"  {len(allx)} _extra_state keys, {len(keys)} matched FP8_LAYERS")
        for k in allx[:6]:
            print("    seen:", k)
    if not keys:
        return []

    if dump:
        keys = keys[:1]  # one key is enough to show the layout, and
        # avoids reading 256 entries off 2048 shards
    sd = {k: io.BytesIO() for k in keys}
    dcp.load(sd, storage_reader=reader)

    rows = []
    for k in keys:
        try:
            d = decode(sd[k])
        except Exception as e:
            if dump:
                b = sd[k]
                t = b[0] if isinstance(b, list) else b
                n = t.numel() if hasattr(t, "numel") else -1
                print(
                    f"    decode failed {k.split('/')[0].split('.')[-2]}: "
                    f"{type(e).__name__} (payload {n} bytes): {e}"
                )
            continue
        if not isinstance(d, dict):
            if dump:
                print(f"    {k} decoded to {type(d)} (not a dict): {repr(d)[:300]}")
                return DUMPED
            continue
        if dump:
            print(f"--- {k}")
            for kk, vv in flatten(d).items():
                print(f"    {kk}: {tuple(vv.shape)} min={vv.min():.6g} max={vv.max():.6g}")
            return DUMPED

        layer = k.split("._extra_state")[0].replace("decoder.layers.", "")
        shard = int(re.search(r"shard_(\d+)_", k).group(1))

        # TE delayed scaling. Column 0 of the fwd triple is the GEMM INPUT --
        # i.e. the activation. Column 1 is the weight. Keeping them apart matters:
        # a "median scale_fwd" over all three blends activations with weights.
        hist = d.get("amax_history_fwd")
        scale = d.get("scale_fwd")
        if hist is None or scale is None:
            continue
        hist, scale = hist.float(), scale.float()

        rows.append(
            {
                "layer": layer,
                "shard": shard,
                # max over the 1024-step rolling window -- what the recipe actually
                # sees when it picks the scale.
                "amax_act": hist[:, 0].max().item(),
                "amax_wgt": hist[:, 1].max().item() if hist.shape[1] > 1 else float("nan"),
                "amax_act_last": hist[0, 0].item(),
                "scale_act": scale[0].item(),
                "scale_fwd_med": scale.median().item(),
            }
        )
    return rows


def _fmt(row):
    """Trim float noise at write time.

    These come from bf16 tensors upcast to float32, so 17-digit repr is
    pure expansion noise -- 6 significant figures is far more than the
    source has, and keeps the committed CSVs under the 500 KB pre-commit
    limit. Formatting happens here, not in the row dicts, so in-memory
    arithmetic is unaffected.
    """
    return {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_root", type=Path)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--dump", action="store_true", help="print one entry's layout and stop")
    args = ap.parse_args()

    ckpts = sorted(
        args.ckpt_root.glob("iter_*"), key=lambda p: int(re.search(r"\d+", p.name).group())
    )

    all_rows = []
    dumped = False
    header = [None]
    for ckpt in ckpts:
        it = int(re.search(r"\d+", ckpt.name).group())
        if args.dump:
            print(f"== {ckpt.name}")
        try:
            rows = scan(ckpt, dump=args.dump)
            dumped = args.dump and rows is DUMPED
        except Exception as e:
            print(f"{it:<8} skipped: {type(e).__name__}")
            continue
        if args.dump and dumped:
            return
        if rows is DUMPED:
            continue
        for r in rows:
            r["iter"] = it
        all_rows += rows

        # A checkpoint whose _extra_state would not decode yields no rows. That
        # is a RESULT, not a reason to abort the sweep -- keep going so the rest
        # of the history still gets read, and say so loudly.
        if not rows:
            print(f"{it:<9} no FP8 state decoded")
            continue

        # Median across the 64 layers, per module type. Falling scale / rising
        # amax = activations growing.
        mods = header[0] or sorted({r["layer"] for r in rows})

        def med(mod, f):
            v = sorted(r[f] for r in rows if r["layer"] == mod)
            return v[len(v) // 2] if v else float("nan")

        if header[0] is None:
            header[0] = mods
            print(
                "iter      "
                + "  ".join(f"{m.split('.')[-1]:>13}" for m in mods)
                + "   | median scale_fwd"
            )
        allsc = sorted(r["scale_fwd_med"] for r in rows)
        print(
            f"{it:<9} "
            + "  ".join(f"{med(m, 'amax_act'):>13.4f}" for m in mods)
            + f"   | {allsc[len(allsc) // 2]:>10.1f}"
        )

    if args.csv and all_rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(_fmt(r) for r in all_rows)
        print(f"\nwrote {len(all_rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
