#!/usr/bin/env python3
"""Has the model lost functional capacity, or only momentum?

WHY THIS EXISTS
---------------
Every LR reduction tried on the flagship halts the loss rise without restoring
descent: the best sustained rate across three interventions is about -0.0005 per
1,000 iterations against a healthy -0.001 to -0.002 (items 39 and the revival).
Edge of stability predicts that dropping below the threshold restores descent. It
did not. That points at the competing story -- the model can no longer move
usefully -- and item 41 supplies a mechanism: `fc2` and `proj` grow as t^0.65, so
their effective LR has fallen 4.2x from 8k to 75k while `qkv`/`fc1` fell 1.8x.

The two stories prescribe OPPOSITE remedies, so they have to be separated before
any restart recipe is chosen. Item 24 is the test and its activation half has
never been run: DEBUG.md has weight-side proxies only, and a neuron can be
functionally dead while its weight-row norm is perfectly healthy.

WHAT IS MEASURED
----------------
Two things, on real packed production batches, with the block-diagonal mask the
run actually trains under:

  residual-stream effective rank   how many directions the representation uses.
                                   Reported three ways because they weight the
                                   spectrum tail differently -- participation
                                   ratio, stable rank, and entropy rank. Collapse
                                   in these is loss of representational capacity.

  SwiGLU per-neuron activity       RMS of each neuron's post-activation output
                                   over tokens, taken at the input of
                                   `linear_fc2`, which under a gated linear unit
                                   IS the gate*up product. A neuron whose RMS is
                                   a thousandth of the layer median contributes
                                   nothing regardless of its weights.

The dead-neuron threshold is relative to each layer's own median neuron, not
absolute, so it stays comparable across checkpoints whose activation scale has
drifted -- and it has drifted: item 14 reports `linear_fc2` max/RMS going 6.0 to
48.2.

WHY CENTERED COVARIANCE
-----------------------
The residual stream carries a large common mean component. An uncentered second
moment is dominated by it and would report a healthy rank for a representation
that is one fixed vector plus noise. The mean is removed, so the rank counts
directions the representation actually varies along.

  APPTAINERENV_LD_LIBRARY_PATH=<stub>:/usr/local/lib \\
  apptainer exec --nv <sif> python3 scripts/scan_activation_capacity.py \\
      <ckpt_root>/iter_0064000 --config <frozen config-*.yaml> \\
      --datamix /e/project1/.../flagship_datamix_option5_fscratch.txt \\
      --csv docs/64k-debug/data/activation_capacity.csv

Several checkpoints may be given, in which case their weight AVERAGE is scored,
exactly as in scan_loss_breakdown.py.

One GPU, ~64 GB of weights. Sources and windows are fixed by seed, so two
checkpoints see identical tokens and the comparison is a difference, not a
sample.
"""

import argparse
import csv
import os
import re
import socket
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from scan_attention_entropy import build_config, load_weights_avg  # noqa: E402
from scan_loss_breakdown import build_mask, packed_window, sources_from_datamix  # noqa: E402


def effective_ranks(cov):
    """Three spectrum summaries of a covariance matrix.

    They disagree on purpose. Participation ratio is the standard
    "effective number of dimensions" and is dominated by the bulk;
    stable rank is set almost entirely by the leading eigenvalue;
    entropy rank sits between them. A representation that is collapsing
    along its tail moves them at different rates, and reporting one
    alone would hide that.
    """
    ev = torch.linalg.eigvalsh(cov.double())
    ev = ev.clamp_min(0)
    s1 = ev.sum()
    if s1 <= 0:
        return float("nan"), float("nan"), float("nan")
    s2 = (ev * ev).sum()
    p = (ev / s1).clamp_min(1e-30)
    return (
        (s1 * s1 / s2).item(),  # participation ratio
        (s1 / ev.max()).item(),  # stable rank
        torch.exp(-(p * p.log()).sum()).item(),  # entropy rank
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path, nargs="+", help="iter_* dir(s); several are averaged")
    ap.add_argument("--config", required=True)
    ap.add_argument("--datamix", required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--windows", type=int, default=8, help="packed sequences per source")
    ap.add_argument("--top", type=int, default=4, help="sources, heaviest first")
    ap.add_argument("--doc-offset", type=int, default=0)
    ap.add_argument(
        "--layer-stride",
        type=int,
        default=8,
        help="residual covariance every Nth layer; SwiGLU stats are taken on all",
    )
    ap.add_argument(
        "--dead-frac",
        type=float,
        default=1e-3,
        help="a neuron is dead if its RMS is below this fraction of the layer median",
    )
    args = ap.parse_args()

    # A world_size=1 process group talking to itself, so no peer has to agree on
    # the port -- while on a shared login node an inherited MASTER_PORT is very
    # likely already held and would abort before a weight is read.
    with socket.socket() as _s:
        _s.bind(("127.0.0.1", 0))
        os.environ["MASTER_PORT"] = str(_s.getsockname()[1])
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    from megatron.core import parallel_state
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.transformer.enums import AttnMaskType

    torch.distributed.init_process_group("nccl", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(1, 1)

    cfg, leaf = build_config(args.config, args.seq_len)
    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=None, moe_grouped_gemm=False, qk_layernorm=cfg.qk_layernorm
    )
    # The spec hardcodes AttnMaskType.causal, under which Transformer Engine
    # builds the mask itself and IGNORES the tensor passed to forward(). The
    # block-diagonal mask this run trains with would then be silently discarded.
    spec.submodules.self_attention.params["attn_mask_type"] = AttnMaskType.arbitrary
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
    load_weights_avg(model, args.ckpt)

    iters = [int(re.search(r"\d+", c.name).group()) for c in args.ckpt]
    it = sum(iters) // len(iters)
    tag = "+".join(str(i) for i in iters) if len(iters) > 1 else str(it)

    layers = list(model.decoder.layers)
    nl = len(layers)
    cov_layers = sorted({0, nl - 1} | set(range(0, nl, args.layer_stride)))
    hidden = cfg.hidden_size
    print(f"{nl} layers; residual covariance on {len(cov_layers)}: {cov_layers}", flush=True)

    # Accumulators live in fp64 on the GPU: a 5120x5120 second moment summed over
    # ~130k tokens loses meaningful precision in fp32, and the whole point is the
    # small eigenvalues in the tail.
    n_tok = 0
    sum_x = {i: torch.zeros(hidden, dtype=torch.float64, device="cuda") for i in cov_layers}
    sum_xx = {
        i: torch.zeros(hidden, hidden, dtype=torch.float64, device="cuda") for i in cov_layers
    }
    ffn = None
    sum_a2 = {}

    handles = []

    def resid_hook(idx):
        def fn(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            x = h.reshape(-1, h.shape[-1]).double()  # [s*b, hidden]
            sum_x[idx] += x.sum(0)
            sum_xx[idx] += x.T @ x

        return fn

    def swiglu_hook(idx):
        def fn(_mod, inp):
            a = inp[0]
            a = a.reshape(-1, a.shape[-1]).double()  # [s*b, ffn_hidden]
            nonlocal ffn
            if ffn is None:
                ffn = a.shape[-1]
            if idx not in sum_a2:
                sum_a2[idx] = torch.zeros(a.shape[-1], dtype=torch.float64, device="cuda")
            sum_a2[idx] += (a * a).sum(0)

        return fn

    for i, layer in enumerate(layers):
        if i in cov_layers:
            handles.append(layer.register_forward_hook(resid_hook(i)))
        handles.append(layer.mlp.linear_fc2.register_forward_pre_hook(swiglu_hook(i)))

    pos_ids = torch.arange(args.seq_len, device="cuda").unsqueeze(0)
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    for src, prefix, _w in sources_from_datamix(args.datamix, args.top):
        try:
            ds = IndexedDataset(str(prefix))
        except Exception as e:
            print(f"  {src:<38} SKIP ({type(e).__name__})", flush=True)
            continue
        nxt = args.doc_offset
        done = 0
        while done < args.windows:
            w = packed_window(ds, args.seq_len, nxt)
            if w is None:
                break
            toks, docid, _dpos = w[:3]
            nxt = w[3]
            inp = toks[:-1].view(1, -1).cuda()
            mask = build_mask(docid[:-1], "block", "cuda")
            with torch.no_grad():
                model(inp, pos_ids, mask)
            n_tok += args.seq_len
            done += 1
        print(f"  {src:<38} {done} windows, {n_tok} tokens so far", flush=True)

    for h in handles:
        h.remove()
    if n_tok == 0:
        sys.exit("no tokens scored")

    rows = []
    for i in range(nl):
        rms = (sum_a2[i] / n_tok).sqrt()
        med = rms.median()
        dead = (rms < args.dead_frac * med).double().mean().item()
        q = torch.quantile(rms, torch.tensor([0.01, 0.5, 0.99], dtype=torch.float64, device="cuda"))
        row = {
            "iter": it,
            "ckpt": tag,
            "layer": i,
            "n_tokens": n_tok,
            "ffn_hidden": ffn,
            "swiglu_dead_frac": f"{dead:.6f}",
            "swiglu_rms_p01": f"{q[0].item():.6e}",
            "swiglu_rms_p50": f"{q[1].item():.6e}",
            "swiglu_rms_p99": f"{q[2].item():.6e}",
        }
        if i in cov_layers:
            mu = sum_x[i] / n_tok
            cov = sum_xx[i] / n_tok - torch.outer(mu, mu)
            pr, sr, er = effective_ranks(cov)
            row.update(
                hidden=hidden,
                pr_rank=f"{pr:.3f}",
                stable_rank=f"{sr:.3f}",
                entropy_rank=f"{er:.3f}",
                pr_frac=f"{pr / hidden:.6f}",
            )
        else:
            row.update(hidden=hidden, pr_rank="", stable_rank="", entropy_rank="", pr_frac="")
        rows.append(row)
        if i in cov_layers:
            print(
                f"  layer {i:>2}  PR {row['pr_rank']:>9}  stable {row['stable_rank']:>8}  "
                f"entropy {row['entropy_rank']:>9}  dead {dead * 100:.3f}%",
                flush=True,
            )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    new = not args.csv.exists()
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"-> {args.csv}")


if __name__ == "__main__":
    main()
