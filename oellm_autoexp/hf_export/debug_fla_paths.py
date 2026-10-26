"""Debug: run one FLA (CKDA) layer of an HF export on the input it gets for a given document,
through each kernel path, and compare with the fp32 naive recurrence (NaN positions, max diff).

    python -m oellm_autoexp.hf_export.debug_fla_paths --model-dir <hf> --datamix <mix> --doc 2 --layer 21
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .score_on_flag import layer_device_map, sample_docs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--datamix", type=Path, required=True)
    p.add_argument("--doc", type=int, default=2)
    p.add_argument("--layer", type=int, default=21)
    p.add_argument("--num-docs", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--min-len", type=int, default=256)
    p.add_argument("--gpu-mem-gib", type=float, default=56)
    p.add_argument("--warm-docs", type=int, nargs="*", default=[],
                   help="documents forwarded first (scoring order: Triton autotunes on the first shapes it sees)")
    a = p.parse_args()
    from fla.models.utils import Cache
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(a.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, trust_remote_code=True, dtype=torch.bfloat16,
                                                 device_map=layer_device_map(cfg, a.gpu_mem_gib)).eval()
    docs = sample_docs(a.datamix, a.num_docs, a.min_len, 0)
    d = docs[a.doc]
    dev = next(model.parameters()).device
    from transformers import AutoTokenizer

    bos = getattr(cfg, "bos_token_id", None)
    bos = AutoTokenizer.from_pretrained(a.model_dir).bos_token_id if bos is None else bos
    ids = torch.tensor(np.concatenate([[bos], d[: a.seq_len - 1]]), device=dev).unsqueeze(0)
    for w in a.warm_docs:
        wid = torch.tensor(np.concatenate([[bos], docs[w][: a.seq_len - 1]]), device=dev).unsqueeze(0)
        with torch.no_grad():
            hs = model.model(input_ids=wid, use_cache=False).last_hidden_state
        print(f"warm doc {w}: {wid.shape[1]} tokens, finite {bool(torch.isfinite(hs).all())}", flush=True)
    fla = model.model.layers[a.layer].mixer.layer
    grab = {}
    def keep_input(_mod, _args, kw, _out):  # returns None: a returned value would replace the output
        grab.setdefault("x", kw["hidden_states"].detach().clone())

    h = fla.register_forward_hook(keep_input, with_kwargs=True)
    with torch.no_grad():
        hs = model.model(input_ids=ids, use_cache=False).last_hidden_state
    h.remove()
    print(f"doc {a.doc} in the model: finite {bool(torch.isfinite(hs).all())}", flush=True)
    x = grab["x"]
    T = x.shape[1]
    print(f"layer {a.layer} ({type(fla).__name__}), input {tuple(x.shape)} {x.dtype}, |x| max {x.abs().max().item():.3g}")
    cu = torch.tensor([0, T], device=x.device, dtype=torch.int32)

    def run(name, **kw):
        with torch.no_grad():
            o = fla(hidden_states=x, **kw)[0].float()
        bad = (~torch.isfinite(o)).reshape(o.shape[0], T, -1).any(-1)[0]
        pos = torch.nonzero(bad).flatten()
        print(f"{name:34s} non-finite positions {int(bad.sum()):5d}, first {pos[:3].tolist()}", flush=True)
        return o

    outs = {
        "chunk, use_cache (HF wrapper)": run("chunk, use_cache (HF wrapper)", past_key_values=Cache(), use_cache=True),
        "chunk, no cache, no cu_seqlens": run("chunk, no cache, no cu_seqlens", use_cache=False),
        "chunk, no cache, cu_seqlens (train)": run("chunk, no cache, cu_seqlens (train)", use_cache=False, cu_seqlens=cu),
        "chunk, use_cache, cu_seqlens": run("chunk, use_cache, cu_seqlens", past_key_values=Cache(), use_cache=True,
                                             cu_seqlens=cu),
    }
    fla.backend = "naive_recurrent"
    ref = run("naive recurrent (bf16 in, ref)", use_cache=False)
    base = outs["chunk, no cache, cu_seqlens (train)"]
    for n, o in outs.items():
        ok = torch.isfinite(o) & torch.isfinite(base)
        print(f"{n:34s} max |o - train path| {float((o - base).abs()[ok].max()) if ok.any() else float('nan'):.3g}")
    for n, o in outs.items():
        ok = torch.isfinite(o) & torch.isfinite(ref)
        print(f"{n:34s} max |o - ref| on finite {float((o - ref).abs()[ok].max()):.3g}, |ref| max {float(ref[ok].abs().max()):.3g}")


if __name__ == "__main__":
    main()
