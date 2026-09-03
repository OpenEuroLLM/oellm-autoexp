#!/usr/bin/env python3
"""Per-layer residual-stream budget: how much does each sublayer actually add?

The gain plots cannot answer whether a layer has stopped working, because a
RMSNorm gain and the weight it feeds are a degenerate pair -- only the product
`gain x W` affects the function, so a shrinking gain paired with a growing weight
is pure bookkeeping (item 14). The quantity that IS gauge-invariant is the size of
what the branch writes into the residual stream, relative to the stream itself:

    attn_ratio = rms( Attn(RMSNorm(h) * g) ) / rms(h)

That is measured here, with hooks on a real forward pass, rather than inferred
from weight norms. The static proxy used in item 30 (rms(g) x rms(W_qkv) x
rms(W_proj)) is weak: the RMS of a matrix product is not the product of the RMSs
when the weights are correlated or the input is anisotropic, and it cannot see
what the softmax does.

Two questions it answers, in order of interest:

 1. **Model-wide.** If every layer's ratio falls together -- the residual stream
    growing faster than the branches feeding it -- that is genuine functional
    collapse across the whole model, and it is untested. This is the reason to
    run the scan.
 2. **Layers 0-1.** How much of the 25x gain collapse at layer 0 is real. Expect
    a footnote confirming item 30's ~2.8x, not a trigger: 86% of layer 0's
    decline is complete by 60,000, before the onset.

Also records the cosine between each branch's output and the residual stream it
is added to, which separates "the branch writes new information" from "the branch
just rescales what is already there".

⚠️ Read this across checkpoints, never one at a time. The ratio falls with depth
in every trained transformer -- that is the baseline shape, not a symptom. Only a
change in the shape over training means anything. And do not look for a kink at
64,000: the loss departs at ~60,500 and grows smoothly (item 27), so this scan
answers *how much* function was lost, not *when*.

One checkpoint per invocation, ~64 GB of weights on the GPU; fits on a login node.

  for it in 8000 34454 60000 64000 75126; do
    APPTAINERENV_LD_LIBRARY_PATH=<stub>:/usr/local/lib \
    apptainer exec --nv <sif> python3 scripts/scan_residual_stream.py \
        <ckpt_root>/iter_$(printf %07d $it) --config <frozen config-*.yaml> \
        --data <shard prefix> --csv docs/64k-debug/data/residual_stream.csv
  done
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_attention_entropy import (  # noqa: E402
    build_config,
    fixed_batch,
    load_weights,
    verify_load,
)


def rms(t):
    return t.float().pow(2).mean().sqrt().item()


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def first(*objs):
    """The first tensor in here.

    Sublayers return (output, bias); the residual stream is a bare
    tensor; and TransformerLayer is called with `hidden_states=` as a
    keyword, so a positional-only hook sees an empty tuple.
    """
    for o in objs:
        if isinstance(o, torch.Tensor):
            return o
        if isinstance(o, dict):
            for k in ("hidden_states", "x", "input"):
                if isinstance(o.get(k), torch.Tensor):
                    return o[k]
            o = tuple(o.values())
        if isinstance(o, (tuple, list)):
            t = first(*o)
            if t is not None:
                return t
    return None


def attach(model):
    """Hook the five points that bracket the two branches of every layer.

    Megatron's TransformerLayer.forward is     residual = h <- pre-hook
    on the layer     attn_out = self_attention(norm(h)) <- hook on
    self_attention     h = residual + attn_out     residual = h <- pre-
    hook on pre_mlp_layernorm     mlp_out = mlp(norm(h)) <- hook on mlp
    h = residual + mlp_out            <- hook on the layer
    """
    rows, live, handles = {}, {}, []

    def row(i):
        return rows.setdefault(i, {"layer": i})

    def mk(idx, what):
        def pre(_m, args, kwargs):
            t = first(args, kwargs)
            if t is None:
                return
            live[(idx, what)] = t.detach()
            row(idx)[f"{what}_rms"] = rms(t)

        def post(_m, _args, out):
            t = first(out)
            if t is None:
                return
            r = row(idx)
            if what == "attn":
                r["attn_rms"] = rms(t)
                h = live.pop((idx, "h_in"), None)
                if h is not None and h.shape == t.shape:
                    r["attn_cos"] = cos(t, h)
            elif what == "mlp":
                r["mlp_rms"] = rms(t)
                h = live.pop((idx, "h_mid"), None)
                if h is not None and h.shape == t.shape:
                    r["mlp_cos"] = cos(t, h)
            else:
                r["h_out_rms"] = rms(t)

        return pre, post

    for i, layer in enumerate(model.decoder.layers):
        pre_in, post_out = mk(i, "h_in"), mk(i, "h_out")
        handles.append(layer.register_forward_pre_hook(pre_in[0], with_kwargs=True))
        handles.append(layer.register_forward_hook(post_out[1]))
        handles.append(layer.self_attention.register_forward_hook(mk(i, "attn")[1]))
        handles.append(layer.mlp.register_forward_hook(mk(i, "mlp")[1]))
        # pre_mlp_layernorm is an IdentityOp when the norm is fused into
        # linear_fc1, but it is still called, so the pre-hook fires either way.
        handles.append(
            layer.pre_mlp_layernorm.register_forward_pre_hook(mk(i, "h_mid")[0], with_kwargs=True)
        )
    return rows, handles


GAIN_KEYS = {
    "attn_gain_rms": ("self_attention.linear_qkv.layer_norm_weight", "input_layernorm.weight"),
    "mlp_gain_rms": ("mlp.linear_fc1.layer_norm_weight", "pre_mlp_layernorm.weight"),
}


def add_gains(model, rows):
    """The gain itself, so the four quantities live in one file."""
    sd = model.state_dict()
    for i in rows:
        for col, cands in GAIN_KEYS.items():
            for c in cands:
                k = f"decoder.layers.{i}.{c}"
                if k in sd:
                    rows[i][col] = rms(sd[k])
                    break


COLS = [
    "iter",
    "layer",
    "h_in_rms",
    "attn_rms",
    "attn_ratio",
    "attn_cos",
    "h_mid_rms",
    "mlp_rms",
    "mlp_ratio",
    "mlp_cos",
    "h_out_rms",
    "attn_gain_rms",
    "mlp_gain_rms",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--config", required=True, help="the run's frozen config-*.yaml")
    ap.add_argument("--data", required=True, help="megatron shard prefix (no .bin)")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--doc-offset", type=int, default=0)
    args = ap.parse_args()

    import os

    for k, v in [
        ("MASTER_ADDR", "127.0.0.1"),
        ("MASTER_PORT", "29519"),
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("LOCAL_RANK", "0"),
    ]:
        os.environ.setdefault(k, v)
    torch.distributed.init_process_group("nccl", rank=0, world_size=1)

    from megatron.core import parallel_state as ps
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )

    ps.initialize_model_parallel(1, 1)
    torch.manual_seed(0)

    cfg, leaf = build_config(args.config, args.seq_len)
    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=None, moe_grouped_gemm=False, qk_layernorm=cfg.qk_layernorm
    )
    model = (
        GPTModel(
            config=cfg,
            transformer_layer_spec=spec,
            vocab_size=leaf["padded_vocab_size"],
            max_sequence_length=args.seq_len,
            pre_process=True,
            post_process=True,
            share_embeddings_and_output_weights=not leaf["untie_embeddings_and_output_weights"],
            position_embedding_type=leaf["position_embedding_type"],
            rotary_base=leaf["rotary_base"],
        )
        .cuda()
        .eval()
    )

    load_weights(model, args.ckpt)
    it = int(re.search(r"\d+", args.ckpt.name).group())
    verify_load(model, "docs/64k-debug/data/norm_gains.csv", it)

    tokens, _ = fixed_batch(args.data, args.seq_len, args.batch, args.doc_offset)
    pos = torch.arange(args.seq_len, device=tokens.device).unsqueeze(0)
    mask = torch.triu(
        torch.ones(args.seq_len, args.seq_len, device=tokens.device, dtype=torch.bool), 1
    ).view(1, 1, args.seq_len, args.seq_len)

    rows, handles = attach(model)
    with torch.no_grad():
        model(tokens, pos, mask)
    for h in handles:
        h.remove()
    add_gains(model, rows)

    for r in rows.values():
        if r.get("h_in_rms"):
            r["attn_ratio"] = r["attn_rms"] / r["h_in_rms"]
        if r.get("h_mid_rms"):
            r["mlp_ratio"] = r["mlp_rms"] / r["h_mid_rms"]

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    new = not args.csv.exists()
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        for i in sorted(rows):
            r = rows[i]
            w.writerow(
                {c: (f"{r[c]:.6g}" if isinstance(r.get(c), float) else r.get(c, "")) for c in COLS}
                | {"iter": it}
            )

    ar = [rows[i]["attn_ratio"] for i in sorted(rows) if "attn_ratio" in rows[i]]
    mr = [rows[i]["mlp_ratio"] for i in sorted(rows) if "mlp_ratio" in rows[i]]
    print(
        f"{it}: {len(rows)} layers | attn_ratio L0={ar[0]:.4f} "
        f"median={sorted(ar)[len(ar) // 2]:.4f} | mlp_ratio median="
        f"{sorted(mr)[len(mr) // 2]:.4f} | h_out_rms L0={rows[0]['h_out_rms']:.3f} "
        f"L{len(rows) - 1}={rows[len(rows) - 1]['h_out_rms']:.3f} -> {args.csv}"
    )


if __name__ == "__main__":
    main()
