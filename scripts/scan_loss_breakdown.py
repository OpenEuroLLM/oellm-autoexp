#!/usr/bin/env python3
"""Where does the extra loss actually land? Per data source, and per position
within a document.

WHY THIS EXISTS
---------------
Every measurement in docs/64k-debug/DEBUG.md is loss against ITERATION. Nothing
has measured loss against WHAT IS BEING PREDICTED. That leaves the two most
natural explanations untested and mutually indistinguishable (items 5 and 10):

  * a data-composition effect -- one source becomes harder or is over-exposed.
    Signature: the 64k -> later degradation is concentrated in a few sources.
  * a packed-document / masking effect. Signature: the degradation is
    concentrated at low positions within a document, where the objective depends
    on the mask being right.

If instead the degradation is uniform across sources AND flat across document
position, both are excluded and the cause is model-wide, which is what the
optimization-regime hypotheses predict.

This is the cheap discriminator: no training, one model load per checkpoint.

PAIRING IS THE WHOLE DESIGN
---------------------------
Absolute loss differs between sources by far more than the effect being chased,
so cross-source comparison is meaningless. Every number here is only useful as a
DIFFERENCE between checkpoints on IDENTICAL tokens. Token selection is therefore
fully determined by --datamix, --batch, --windows and --doc-offset, with no
sampling: run the same command against two checkpoints and the token sets match
exactly. Per-token losses are highly correlated across checkpoints, so the
paired standard error is much smaller than the spread of the loss itself.

MASK MODES
----------
`--mask both` runs each token set twice:

  block   block-diagonal per document, so a token never attends across a
          document boundary. This is the production objective
          (`dataloader_inter_document_masking: true`, on since iteration
          34,455).
  dense   plain causal over the concatenated stream, i.e. cross-document
          attention allowed. This is what the flagship trained under BEFORE the
          stack swap, and what the other scan_*.py scripts in this directory
          use.

The block-minus-dense difference on identical tokens is a direct read on how
much the objective change is worth, which item 10 has never isolated.

MEMORY
------
Loss needs the full logits: seq x vocab x 2 bytes, 2.1 GB at 4096 x 262144, and
cross-entropy upcasts to fp32 on top. With ~64 GB of weights already resident
that caps the live batch at 1-2 sequences, so token count is raised with
--windows (sequential forward passes) rather than with --batch.

  APPTAINERENV_LD_LIBRARY_PATH=<stub>:/usr/local/lib \\
  apptainer exec --nv <sif> python3 scripts/scan_loss_breakdown.py \\
      <ckpt_root>/iter_0064000 --config <frozen config-*.yaml> \\
      --datamix /e/project1/.../flagship_datamix_option5_fscratch.txt \\
      --csv docs/64k-debug/data/loss_breakdown.csv

One GPU, fits on a login node. About 25 sources x --windows forward passes.
"""

import argparse
import csv
import os
import re
import socket
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_attention_entropy import build_config, load_weights  # noqa: E402

# Position-within-document bins for the label token. Low bins are where a
# masking fault would show: with the block mask a token at position 0 has only
# itself to attend to, while under the dense mask it sees the previous document.
POS_BINS = [(1, 8), (9, 32), (33, 128), (129, 512), (513, 2048), (2049, 10**9)]


def sources_from_datamix(path, top):
    """Group the blend by source directory, heaviest first.

    Returns [(source, representative_prefix, summed_weight)]. The
    representative is that source's heaviest single shard, so the batch
    comes from the part of the source the model actually saw most of.
    """
    agg = defaultdict(lambda: [0.0, None, -1.0])
    for line in open(path):
        parts = line.split()
        if len(parts) != 2:
            continue
        w, prefix = float(parts[0]), parts[1]
        if "/collection/flag/" not in prefix:
            continue
        src = prefix.split("/collection/flag/")[1].split("/")[0]
        a = agg[src]
        a[0] += w
        if w > a[2]:
            a[1], a[2] = prefix, w
    ranked = sorted(agg.items(), key=lambda kv: -kv[1][0])
    return [(s, v[1], v[0]) for s, v in ranked[:top]]


def packed_window(ds, seq_len, start_doc):
    """One packed sequence, plus the document structure needed for the mask.

    Concatenates whole documents from `start_doc` until seq_len+1 tokens are
    available, exactly as the training dataloader packs them. Returns the token
    window, a per-token document id, a per-token 1-based position within its
    document, and the next document index to start from.
    """
    toks, docid, dpos = [], [], []
    i = start_doc
    while len(toks) < seq_len + 1 and i < len(ds):
        d = ds[i].tolist()
        toks.extend(d)
        docid.extend([i] * len(d))
        dpos.extend(range(1, len(d) + 1))
        i += 1
    if len(toks) < seq_len + 1:
        return None
    n = seq_len + 1
    return (
        torch.tensor(toks[:n], dtype=torch.long),
        torch.tensor(docid[:n], dtype=torch.long),
        torch.tensor(dpos[:n], dtype=torch.long),
        i,
    )


def build_mask(docid_in, mode, device):
    """Megatron convention: True means DO NOT attend."""
    s = docid_in.numel()
    causal = torch.triu(torch.ones(s, s, dtype=torch.bool, device=device), 1)
    if mode == "dense":
        return causal.view(1, 1, s, s)
    d = docid_in.to(device)
    cross = d.unsqueeze(1) != d.unsqueeze(0)
    return (causal | cross).view(1, 1, s, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--config", required=True, help="the run's frozen config-<job>.yaml")
    ap.add_argument("--datamix", required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--windows", type=int, default=8, help="packed sequences per source")
    ap.add_argument("--doc-offset", type=int, default=0)
    ap.add_argument("--top", type=int, default=25, help="sources, heaviest first")
    ap.add_argument(
        "--dump",
        type=Path,
        help="also write every per-token loss to this .npz, for paired analysis",
    )
    ap.add_argument(
        "--entropy",
        action="store_true",
        help="also record per-token predictive entropy and top probability",
    )
    ap.add_argument("--mask", choices=["block", "dense", "both"], default="both")
    args = ap.parse_args()

    for k, v in (
        ("MASTER_ADDR", "127.0.0.1"),
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("LOCAL_RANK", "0"),
    ):
        os.environ.setdefault(k, v)
    # Always claim a fresh port rather than honouring an inherited MASTER_PORT.
    # This is a world_size=1 process group talking to itself, so no peer needs
    # to agree on the number -- while on a shared login node an inherited port
    # is very likely already held by somebody else's job, which aborts the run
    # with EADDRINUSE before a single weight is read.
    with socket.socket() as _s:
        _s.bind(("127.0.0.1", 0))
        os.environ["MASTER_PORT"] = str(_s.getsockname()[1])
    torch.distributed.init_process_group("nccl", rank=0, world_size=1)

    from megatron.core import parallel_state as ps
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.transformer.enums import AttnMaskType

    ps.initialize_model_parallel(1, 1)
    torch.manual_seed(0)

    cfg, leaf = build_config(args.config, args.seq_len)
    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=None, moe_grouped_gemm=False, qk_layernorm=cfg.qk_layernorm
    )
    # The spec hardcodes AttnMaskType.causal, and under `causal` Transformer
    # Engine constructs the mask internally and IGNORES the tensor handed to
    # forward(). Both --mask modes then return bit-identical losses, which reads
    # as "the mask makes no difference" when in fact no mask was ever applied.
    # `arbitrary` is the only mask type that honours a supplied boolean mask. It
    # forces the unfused attention path -- slower and it materialises the score
    # matrix -- but it is the only way this comparison means anything.
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
    load_weights(model, args.ckpt)

    it = int(re.search(r"\d+", args.ckpt.name).group())
    modes = ["block", "dense"] if args.mask == "both" else [args.mask]
    dump = defaultdict(list) if args.dump else None
    pos_ids = torch.arange(args.seq_len, device="cuda").unsqueeze(0)
    rows = []

    for si, (src, prefix, weight) in enumerate(sources_from_datamix(args.datamix, args.top)):
        try:
            ds = IndexedDataset(str(prefix))
        except Exception as e:
            print(f"  {src:<38} SKIP ({type(e).__name__})", flush=True)
            continue

        windows, nxt = [], args.doc_offset
        for _ in range(args.windows):
            w = packed_window(ds, args.seq_len, nxt)
            if w is None:
                break
            windows.append(w[:3])
            nxt = w[3]
        if not windows:
            print(f"  {src:<38} SKIP (too few tokens)", flush=True)
            continue

        for mode in modes:
            tot, ntok = 0.0, 0
            binsum = defaultdict(float)
            binn = defaultdict(int)
            for toks, docid, dpos in windows:
                inp = toks[:-1].view(1, -1).cuda()
                lab = toks[1:].view(1, -1).cuda()
                mask = build_mask(docid[:-1], mode, "cuda")
                with torch.no_grad():
                    if args.entropy:
                        # Predictive entropy is a MARGINAL statistic: no
                        # conditioning on either checkpoint's own loss, so it is
                        # immune to the regression-to-the-mean artefact that
                        # makes a loss-sorted table show flattening even under a
                        # null. Chunked over the sequence because a full
                        # [4096, 262144] fp32 softmax is 4.3 GB per copy.
                        lg = model(inp, pos_ids, mask)  # [b, s, V]
                        parts, ents, mx = [], [], []
                        for c in range(0, lg.shape[1], 512):
                            z = lg[:, c : c + 512].float()
                            lp = torch.log_softmax(z, dim=-1)
                            pr = lp.exp()
                            ents.append((-(pr * lp).sum(-1)).view(-1))
                            mx.append(pr.max(-1).values.view(-1))
                            tgt = lab[:, c : c + 512]
                            parts.append(-lp.gather(-1, tgt.unsqueeze(-1)).view(-1))
                            del z, lp, pr
                        loss = torch.cat(parts).cpu()
                        ent = torch.cat(ents).cpu()
                        top = torch.cat(mx).cpu()
                        del lg, parts, ents, mx
                    else:
                        loss = model(inp, pos_ids, mask, labels=lab).float().view(-1).cpu()
                        ent = top = None
                if dump is not None:
                    # Keep every per-token loss, not just its mean. Two
                    # checkpoints scored on identical tokens can be subtracted
                    # token by token afterwards, which separates "everything got
                    # slightly worse" from "a tail got much worse" -- those have
                    # the same mean and completely different causes.
                    dump["loss"].append(loss.numpy().astype("float32"))
                    dump["dpos"].append(dpos[1:].numpy().astype("int32"))
                    dump["src"].append(np.full(loss.numel(), si, dtype="int16"))
                    dump["mode"].append(np.full(loss.numel(), modes.index(mode), dtype="int8"))
                    if ent is not None:
                        dump["entropy"].append(ent.numpy().astype("float32"))
                        dump["top_prob"].append(top.numpy().astype("float32"))
                tot += loss.sum().item()
                ntok += loss.numel()
                lp = dpos[1:]  # position of the LABEL token in its document
                for lo, hi in POS_BINS:
                    m = (lp >= lo) & (lp <= hi)
                    if m.any():
                        binsum[(lo, hi)] += loss[m].sum().item()
                        binn[(lo, hi)] += int(m.sum())
            row = {
                "iter": it,
                "source": src,
                "mask": mode,
                "weight": f"{weight:.6f}",
                "n_tokens": ntok,
                "mean_loss": f"{tot / ntok:.6f}",
            }
            for lo, hi in POS_BINS:
                k = f"pos_{lo}_{hi if hi < 10**9 else 'inf'}"
                row[k] = f"{binsum[(lo, hi)] / binn[(lo, hi)]:.6f}" if binn[(lo, hi)] else ""
            rows.append(row)
            print(f"  {src:<38} {mode:<6} {ntok:>7} tok  loss {tot / ntok:.5f}", flush=True)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    new = not args.csv.exists()
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"iter {it}: {len(rows)} rows -> {args.csv}")

    if dump is not None:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        names = [s for s, _, _ in sources_from_datamix(args.datamix, args.top)]
        np.savez_compressed(
            args.dump,
            loss=np.concatenate(dump["loss"]),
            dpos=np.concatenate(dump["dpos"]),
            src=np.concatenate(dump["src"]),
            mode=np.concatenate(dump["mode"]),
            source_names=np.array(names),
            mode_names=np.array(modes),
            iteration=np.array(it),
            **{k: np.concatenate(dump[k]) for k in ("entropy", "top_prob") if dump.get(k)},
        )
        print(f"iter {it}: {len(np.concatenate(dump['loss'])):,} token losses -> {args.dump}")


if __name__ == "__main__":
    main()
