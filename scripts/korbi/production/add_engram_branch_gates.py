#!/usr/bin/env python3
"""Add per-stream Engram gates (Megatron ``engram_branch_gates``) to an fsdp_dtensor checkpoint, function-preserving.

Under mHC the Engram layer gates its value once, on the mHC read, and every residual stream takes the same
output. With ``engram_branch_gates`` every stream m has its own gate from its own key projection, key norm and
hidden norm (the Engram paper's multi-branch integration); stream 0 keeps ``key_proj`` / ``key_norm`` /
``hidden_norm``, streams 1 .. n-1 get ``key_proj_extra.<m-1>`` / ``key_norm_extra.<m-1>`` /
``hidden_norm_extra.<m-1>``. Per Engram layer this tool writes those as copies of stream 0's, with copies of
their optimizer state (MuOwn classical g, m_g, v_g, v_norm, m_u for the projection; Adam exp_avg /
exp_avg_sq for the norm gains): with ``engram_branch_query=read`` (or ``learned``, lambda = 0) the continued
model computes what it did (tests/unit_tests/ssm/test_engram_branch_gates.py), and the copies diverge as the
streams' gradients differ. ``engram_branch_query=stream`` changes the gates' queries at once.

  --query-mix    gate_terms.query_mix = 0 (+ zero Adam state), for engram_branch_query=learned
  --gate-scale   gate_terms.scale = 1 (+ zero Adam state), for engram_gate_scale
  --logit-scale  gate_terms.logit_scale = the per-layer value (+ zero Adam state), for
                 engram_gate_logit_scale; init at the checkpoint's mean query RMS (e.g. 2:0.016,52:0.47)

The source's .distcp files are hard-linked (dcp_surgery.write_surgery); only the new tensors are written.

    python add_engram_branch_gates.py --src <ckpt>/iter_N --dst <new>/ckpt/iter_N [--query-mix] [--gate-scale]
    python add_engram_branch_gates.py selftest
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODEL = "model.module.decoder.layers.{l}.{n}"
OPT = "optimizer.0.state.module.module.module.decoder.layers.{l}.{n}.{s}"
STATES = ("g", "m_g", "v_g", "v_norm", "m_u", "exp_avg", "exp_avg_sq")
COPIED = ("key_proj.weight", "hidden_norm.weight", "key_norm.weight")


def engram_layers(smd) -> list[int]:
    return sorted(
        int(m.group(1))
        for k in smd
        for m in [re.fullmatch(r"model\.module\.decoder\.layers\.(\d+)\.key_proj\.weight", k)]
        if m
    )


def source_keys(smd, layer: int) -> dict[tuple[str, str], str]:
    """(parameter, state or "") -> checkpoint key of stream 0's copied tensors of ``layer``."""
    out = {}
    for n in COPIED:
        out[(n, "")] = MODEL.format(l=layer, n=n)
        for s in STATES:
            k = OPT.format(l=layer, n=n, s=s)
            if k in smd:
                out[(n, s)] = k
    for n in COPIED:
        states = {s for (p, s) in out if p == n and s}
        if not states:
            raise SystemExit(f"layer {layer} {n}: no optimizer state in the checkpoint")
    return out


def derive(smd, src: dict[str, torch.Tensor], streams: int, query_mix: bool, gate_scale: bool,
           logit_scale: dict[int, float] | None = None, branch: bool = True,
           logit_scale_default: float | None = None):
    """The new tensors: ``src`` holds the tensors of :func:`source_keys` (by key)."""
    replace = {}
    for layer in engram_layers(smd):
        keys = source_keys(smd, layer)
        if branch:
            for m in range(streams - 1):
                for (n, s), key in keys.items():
                    stem, leaf = n.rsplit(".", 1)
                    new_n = f"{stem}_extra.{m}.{leaf}"
                    new = MODEL.format(l=layer, n=new_n) if not s else OPT.format(l=layer, n=new_n, s=s)
                    if new in smd:
                        raise SystemExit(f"{new} already exists: the source has per-stream gates already")
                    replace[new] = src[key].clone()
        gain = smd[MODEL.format(l=layer, n="hidden_norm.weight")].properties.dtype
        adam = smd[OPT.format(l=layer, n="hidden_norm.weight", s="exp_avg")].properties.dtype
        for flag, name, value in ((query_mix, "query_mix", 0.0), (gate_scale, "scale", 1.0)):
            if not flag:
                continue
            n = f"gate_terms.{name}"
            replace[MODEL.format(l=layer, n=n)] = torch.full((streams,), value, dtype=gain)
            for s in ("exp_avg", "exp_avg_sq"):
                replace[OPT.format(l=layer, n=n, s=s)] = torch.zeros(streams, dtype=adam)
        if logit_scale is not None:
            if layer in logit_scale:
                value = logit_scale[layer]
            elif logit_scale_default is not None:
                value = logit_scale_default
            else:
                raise SystemExit(f"no --logit-scale init for layer {layer} (have {sorted(logit_scale)})")
            n = "gate_terms.logit_scale"
            replace[MODEL.format(l=layer, n=n)] = torch.full((streams,), float(value), dtype=gain)
            for s in ("exp_avg", "exp_avg_sq"):
                replace[OPT.format(l=layer, n=n, s=s)] = torch.zeros(streams, dtype=adam)
    return replace


def parse_logit_scale(text: str) -> dict[int, float]:
    """``"2:0.016,52:0.47"`` -> ``{2: 0.016, 52: 0.47}`` (per-layer init of ``gate_terms.logit_scale``)."""
    out: dict[int, float] = {}
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        layer, value = part.split(":")
        out[int(layer)] = float(value)
    return out


def read(reader, smd, keys) -> dict[str, torch.Tensor]:
    import torch.distributed.checkpoint as dcp

    want = {k: torch.empty(tuple(smd[k].size), dtype=smd[k].properties.dtype) for k in keys}
    dcp.load(want, storage_reader=reader, no_dist=True)
    return want


def apply(a) -> int:
    from torch.distributed.checkpoint import FileSystemReader
    from dcp_surgery import write_surgery

    t0 = time.time()
    log = lambda *x: print(f"[{time.time() - t0:7.1f}s]", *x, flush=True)  # noqa: E731
    reader = FileSystemReader(a.src)
    md = reader.read_metadata()
    smd = md.state_dict_metadata
    layers = engram_layers(smd)
    if not layers:
        raise SystemExit("no Engram layers (decoder.layers.<i>.key_proj.weight) in the checkpoint")
    keys = [k for layer in layers for k in source_keys(smd, layer).values()]
    log(f"Engram layers {layers}; {len(keys)} source tensors")
    replace = derive(smd, read(reader, smd, keys), a.streams, a.query_mix, a.gate_scale,
                     parse_logit_scale(a.logit_scale) if (a.logit_scale or a.logit_scale_default is not None) else None,
                     branch=a.branch, logit_scale_default=a.logit_scale_default)
    for k in sorted(replace):
        log(f"  {k.split('decoder.layers.')[-1]} {tuple(replace[k].shape)} {replace[k].dtype}")
    if a.dry_run:
        log(f"dry run: {len(replace)} new tensors, nothing written")
        return 0
    write_surgery(a.src, a.dst, md, replace, tag="engram_branch", chunk_bytes=64 << 20, log=log)
    back = read(FileSystemReader(a.dst), {**smd, **_meta(a.dst)}, list(replace))
    for k, v in replace.items():
        assert torch.equal(back[k], v), k
    it = re.search(r"iter_(\d+)", os.path.basename(os.path.normpath(a.dst)))
    if it:
        Path(a.dst).parent.joinpath("latest_checkpointed_iteration.txt").write_text(f"{int(it.group(1))}\n")
    log(f"read-back check passed; wrote {a.dst}")
    return 0


def _meta(path):
    from torch.distributed.checkpoint import FileSystemReader

    return FileSystemReader(path).read_metadata().state_dict_metadata


def selftest() -> int:
    """A fake two-Engram-layer checkpoint through the surgery: the new keys are copies of stream 0's."""
    import tempfile

    import torch.distributed.checkpoint as dcp

    gen = torch.Generator().manual_seed(0)
    h, c, n = 8, 6, 4
    state = {}
    for layer in (2, 5):
        state[MODEL.format(l=layer, n="key_proj.weight")] = torch.randn(h, c, generator=gen).bfloat16()
        state[MODEL.format(l=layer, n="value_proj.weight")] = torch.randn(h, c, generator=gen).bfloat16()
        for s, shape in (("g", (h, 1)), ("m_g", (h, 1)), ("v_g", (h, 1)), ("v_norm", (h, 1)), ("m_u", (h, c))):
            state[OPT.format(l=layer, n="key_proj.weight", s=s)] = torch.randn(*shape, generator=gen)
        for nm in ("hidden_norm.weight", "key_norm.weight"):
            state[MODEL.format(l=layer, n=nm)] = torch.rand(h, generator=gen).bfloat16()
            for s in ("exp_avg", "exp_avg_sq"):
                state[OPT.format(l=layer, n=nm, s=s)] = torch.rand(h, generator=gen)
    with tempfile.TemporaryDirectory() as root:
        src, dst = os.path.join(root, "src", "iter_0000010"), os.path.join(root, "dst", "iter_0000010")
        dcp.save(state, storage_writer=dcp.FileSystemWriter(src), no_dist=True)
        rc = apply(argparse.Namespace(src=src, dst=dst, streams=n, query_mix=True, gate_scale=True,
                                      logit_scale="2:0.02,5:0.4", logit_scale_default=None, branch=True,
                                      dry_run=False))
        smd = _meta(dst)
        assert set(state) <= set(smd)
        new = [k for k in smd if k not in state]
        # per layer: 3 tensors x (n - 1) streams with 5 / 2 / 2 states, + 3 scalars x 3
        assert len(new) == 2 * ((n - 1) * (3 + 5 + 2 + 2) + 9), len(new)
        back = read(dcp.FileSystemReader(dst), smd, list(smd))
        for layer in (2, 5):
            for m in range(n - 1):
                for nm, ex in (("key_proj.weight", "key_proj_extra"), ("hidden_norm.weight", "hidden_norm_extra")):
                    leaf = nm.rsplit(".", 1)[1]
                    assert torch.equal(back[MODEL.format(l=layer, n=f"{ex}.{m}.{leaf}")], state[MODEL.format(l=layer, n=nm)])
                assert torch.equal(back[OPT.format(l=layer, n=f"key_proj_extra.{m}.weight", s="m_u")],
                                   state[OPT.format(l=layer, n="key_proj.weight", s="m_u")])
            assert torch.equal(back[MODEL.format(l=layer, n="gate_terms.scale")], torch.ones(n, dtype=torch.bfloat16))
            assert torch.equal(back[MODEL.format(l=layer, n="gate_terms.logit_scale")],
                               torch.full((n,), 0.02 if layer == 2 else 0.4, dtype=torch.bfloat16))
            assert torch.equal(back[MODEL.format(l=layer, n="key_proj.weight")], state[MODEL.format(l=layer, n="key_proj.weight")])
        assert open(os.path.join(root, "dst", "latest_checkpointed_iteration.txt")).read() == "10\n"
    print("selftest passed")
    return rc


def main(argv=None) -> int:
    if (argv if argv is not None else sys.argv[1:])[:1] == ["selftest"]:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--streams", type=int, default=4, help="mhc_expansion_rate")
    ap.add_argument("--query-mix", action="store_true", help="for engram_branch_query=learned")
    ap.add_argument("--gate-scale", action="store_true", help="for engram_gate_scale")
    ap.add_argument("--logit-scale", default="",
                    help="per-layer init of gate_terms.logit_scale, e.g. 2:0.016,52:0.47 (for engram_gate_logit_scale)")
    ap.add_argument("--logit-scale-default", type=float, default=None,
                    help="init of gate_terms.logit_scale for layers not in --logit-scale (e.g. 1.0 on a random init)")
    ap.add_argument("--no-branch", dest="branch", action="store_false",
                    help="only the gate scalars, no per-stream key/norm copies (the fix-only arm)")
    ap.add_argument("--dry-run", action="store_true")
    return apply(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
