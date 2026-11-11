#!/usr/bin/env python3
"""Checkpoint surgery for the native 1.7B ladder checkpoints (torch format, mp_rank_00/model_optim_rng.pt).

Writes a MODEL-ONLY copy of a checkpoint with the tensors an SKDA_INTERVENTION arm adds (interventions.py),
so the native codebase's strict loader needs no change; train it with --no-load-optim --no-load-rng
--finetune and a fresh optimizer (the paired control `ctrl_fresh` is the same copy with no additions).

Added branches (add_attn / add_swa): the hybrid's fla attention class with the arm's kwargs; their q / k /
v / g (and norms) are copied from the NEAREST existing attention layer (--init copy, default) or left at
the class's default init (--init random); o_proj is ZERO, so the model is the original one at step 0 and
the branch's output grows from what it learns. attn_rope adds no tensors.

attn_kv@<n> (GQA reverse test): the existing attention layers' k_proj / v_proj (and k_norm / bias, if the layer has
them) are REPLACED by the per-group means of the original heads -- group g = query heads [g*G, (g+1)*G),
G = num_heads / n, the grouping flash-attn and SDPA(enable_gqa) use -- computed in float64 and cast back. This
changes the function (the insertion loss gap is measured by the smoke); recall_surgery.json lists every replaced
tensor with old / new shape and sha256.

    PYTHONPATH=<code>:<code>/lm_scaling python surgery.py --src <ckpt dir> --dst <new ckpt dir> \
        --intervention "add_attn@lin_late" [--init copy|random] [--seed 1234]

Writes <dst>/iter_<it>/mp_rank_00/model_optim_rng.pt, <dst>/latest_checkpointed_iteration.txt and
<dst>/recall_surgery.json (arguments, source sha256, every new tensor's sha256).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from types import SimpleNamespace


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(t) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().view(-1).view(__import__("torch").uint8).numpy().tobytes()).hexdigest()


def pool_kv(weight, num_heads: int, num_kv: int, head_dim: int):
    """Per-group mean of the heads of a k_proj / v_proj weight (num_heads*head_dim, in) or bias (num_heads*head_dim,)
    -> (num_kv*head_dim, ...): KV head g = mean of the original heads g*G .. (g+1)*G-1 (G = num_heads // num_kv),
    float64 accumulation, original dtype."""
    if num_heads % num_kv:
        raise ValueError(f"{num_heads} heads do not split into {num_kv} groups")
    if weight.shape[0] != num_heads * head_dim:
        raise ValueError(f"{tuple(weight.shape)}: first dim is not {num_heads} x {head_dim}")
    g = num_heads // num_kv
    w = weight.double().reshape(num_kv, g, head_dim, *weight.shape[1:]).mean(dim=1)
    return w.reshape(num_kv * head_dim, *weight.shape[1:]).to(weight.dtype).contiguous()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="checkpoint dir (holds latest_checkpointed_iteration.txt)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--intervention", default="", help="SKDA_INTERVENTION value ('' = ctrl_fresh)")
    ap.add_argument("--init", choices=["copy", "random"], default="copy")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--arm", default="skda-shipped-hybrid-lowrank")
    a = ap.parse_args(argv)

    import torch

    it = int(open(os.path.join(a.src, "latest_checkpointed_iteration.txt")).read().strip())
    src_file = os.path.join(a.src, f"iter_{it:07d}", "mp_rank_00", "model_optim_rng.pt")
    ck = torch.load(src_file, map_location="cpu", weights_only=False)
    sd = ck["model"]
    margs = ck["args"]
    cfg = SimpleNamespace(hidden_size=margs.hidden_size, num_attention_heads=margs.num_attention_heads,
                          kv_channels=margs.kv_channels, num_layers=margs.num_layers,
                          layernorm_epsilon=margs.layernorm_epsilon)

    os.environ["SKDA_INTERVENTION"] = a.intervention
    from lm_scaling.megatron_ext import interventions as iv_mod  # noqa: E402  (the patched code dir)
    from lm_scaling.megatron_ext.specs import HYBRID_EVERY, HYBRID_OF  # noqa: E402

    iv = iv_mod.parse(a.intervention)
    pure = a.arm not in HYBRID_OF  # a pure linear stack: branches take the hybrid sibling's attention flavor
    every = None if pure else HYBRID_EVERY
    branch_arm = {v: k for k, v in HYBRID_OF.items()}[a.arm] if pure else a.arm
    branches = iv_mod.branch_layers(iv, cfg.num_layers, every)
    attn_layers = [] if pure else [n for n in range(1, cfg.num_layers + 1) if n % HYBRID_EVERY == 0]
    if pure and a.init == "copy" and branches:
        raise SystemExit("--init copy needs attention layers to copy from; a pure stack takes --init random")
    torch.manual_seed(a.seed)
    new = {}
    for n, window in sorted(branches.items()):
        cls, kw = iv_mod.branch_kwargs(branch_arm, cfg, n, window, iv["add_rope"])
        mod = cls(**kw)
        bsd = {k: v.detach().clone() for k, v in mod.state_dict().items()}
        if a.init == "copy":
            src_n = min(attn_layers, key=lambda m: (abs(m - n), m))
            pre = f"decoder.layers.{src_n - 1}.self_attention.layer."
            for k in bsd:
                if k.startswith("o_proj"):
                    continue
                if pre + k in sd:
                    bsd[k] = sd[pre + k].detach().clone().to(bsd[k].dtype)
        for k in bsd:
            if k.startswith("o_proj"):
                bsd[k] = torch.zeros_like(bsd[k])
        # dtype reference: an attention o_proj, or (pure stack) the mixer's output projection
        ref = (sd.get(f"decoder.layers.{attn_layers[0] - 1}.self_attention.layer.o_proj.weight") if attn_layers
               else next((v for k, v in sd.items() if k.startswith(f"decoder.layers.{n - 1}.self_attention.layer.")
                          and k.endswith("o_proj.weight")), None))
        for k, v in bsd.items():
            key = f"decoder.layers.{n - 1}.self_attention.branch.{k}"
            if key in sd:
                raise SystemExit(f"{key} already in the source checkpoint")
            new[key] = v.to(ref.dtype if ref is not None and v.is_floating_point() else v.dtype)
    if iv["attn_rope"]:
        # RoPE on the existing attention layers: no weights change, but any (persistent) buffer the rotary
        # module brings must be in the checkpoint for the strict loader
        for n in attn_layers:
            cls, kw = iv_mod.branch_kwargs(a.arm, cfg, n, None, True)
            pre = f"decoder.layers.{n - 1}.self_attention.layer."
            for k, v in cls(**kw).state_dict().items():
                if pre + k not in sd:
                    new[pre + k] = v.detach().clone()
                    sd[pre + k] = new[pre + k]
    replaced = {}
    if iv.get("attn_kv"):
        # GQA by mean pooling on the existing attention layers (only k / v change; query heads, g_proj and o_proj
        # are kept). A k_norm / q_norm acts per head on the projection output and has no head axis: unchanged.
        if pure:
            raise SystemExit("attn_kv needs a hybrid stack")
        nkv = int(iv["attn_kv"])
        for n in attn_layers:
            cls, kw = iv_mod.branch_kwargs(a.arm, cfg, n, None, iv["attn_rope"], nkv)
            ref_mod = cls(**kw)
            heads, hd = ref_mod.num_heads, ref_mod.head_dim
            pre = f"decoder.layers.{n - 1}.self_attention.layer."
            want = {k: tuple(v.shape) for k, v in ref_mod.state_dict().items()}
            for k in ("k_proj.weight", "v_proj.weight", "k_proj.bias", "v_proj.bias"):
                if pre + k not in sd:
                    continue
                old = sd[pre + k]
                newt = pool_kv(old, heads, nkv, hd)
                if tuple(newt.shape) != want[k]:
                    raise SystemExit(f"{pre + k}: pooled {tuple(newt.shape)} != the GQA module's {want[k]}")
                replaced[pre + k] = {"old_shape": list(old.shape), "old_sha256": tensor_sha(old),
                                     "shape": list(newt.shape), "dtype": str(newt.dtype), "sha256": tensor_sha(newt)}
                sd[pre + k] = newt
            # every key of the rebuilt layer must be in the copy with its shape (strict load)
            for k, shp in want.items():
                if pre + k not in sd and pre + k not in new:
                    raise SystemExit(f"{pre + k}: missing for the GQA layer")
                got = tuple((new.get(pre + k) if pre + k in new else sd[pre + k]).shape)
                if got != shp:
                    raise SystemExit(f"{pre + k}: {got} != the GQA module's {shp}")
    sd.update(new)

    out = {"model": sd, "checkpoint_version": ck.get("checkpoint_version", 3.0), "iteration": it, "args": margs}
    dst_dir = os.path.join(a.dst, f"iter_{it:07d}", "mp_rank_00")
    os.makedirs(dst_dir, exist_ok=True)
    torch.save(out, os.path.join(dst_dir, "model_optim_rng.pt"))
    with open(os.path.join(a.dst, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write(str(it))
    manifest = {"args": vars(a), "iteration": it, "source": src_file, "source_sha256": sha256_file(src_file),
                "branches": {str(k): v for k, v in branches.items()}, "attn_rope": iv["attn_rope"],
                "attn_kv": iv.get("attn_kv"), "replaced_tensors": replaced,
                "new_tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": tensor_sha(v)}
                                for k, v in new.items()}}
    with open(os.path.join(a.dst, "recall_surgery.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[surgery] {a.intervention or 'ctrl_fresh'}: iteration {it}, {len(new)} new tensors "
          f"({len(branches)} branches, init {a.init}), {len(replaced)} replaced (attn_kv {iv.get('attn_kv')}) "
          f"-> {a.dst}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
