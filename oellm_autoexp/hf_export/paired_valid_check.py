"""Paired check of an HF export against Megatron: the very samples Megatron's fixed-validation evaluation
scores, scored by the HF model, for the same token-weighted loss.

The samples come from Megatron's own ``GPTDataset``, built with the run's settings (seed, BOS per document,
inter-document masking) for one fixed validation set at ``--seq-len``, the way
``megatron/training/datasets/fixed_valid.py`` builds it. Under inter-document masking Megatron resets every
mixer at each document boundary (attention + RoPE positions, CKDA / GDN state and short convs, Engram
history and conv; LDM is per token), so each document segment of a sample (its ``cu_seqlens``) is scored
here as a sequence of its own. Every label counts (no BOS / EOD masking unless the loss mask says so), and
the loss is sum / tokens over the whole set -- Megatron's ``lm loss validation-<set>``, provided the set's
sample count is a multiple of the evaluation's data-parallel size (otherwise Megatron counts a few samples
twice). Build such a set with ``make_valid_subset --samples``.

    python -m oellm_autoexp.hf_export.paired_valid_check --model-dir <hf> --config <run config.yaml> \\
        --prefix <subset>/all_text_document --seq-len 4096 [--gpu-mem-gib 54 --engram-tables-cpu] --out <json>
"""

from __future__ import annotations

import argparse
import json
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from .export_q35 import megatron_args


def build_dataset(m: dict, prefix: str, seq_len: int, cache: Path):
    from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
    from megatron.core.datasets.utils import compile_helpers
    from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer

    compile_helpers()
    if not m.get("dataloader_inter_document_masking"):
        raise ValueError("expects a run with dataloader_inter_document_masking")
    cfg = GPTDatasetConfig(
        random_seed=m["seed"],
        sequence_length=seq_len,
        blend=None,
        blend_per_split=[None, ([prefix], None), None],
        split=None,
        multiple_validation_sets=True,
        full_validation=True,
        path_to_cache=str(cache),
        mmap_bin_files=m.get("mmap_bin_files", True),
        tokenizer=build_tokenizer(Namespace(**m)),
        reset_position_ids=m["reset_position_ids"],
        reset_attention_mask=m["reset_attention_mask"],
        eod_mask_loss=m["eod_mask_loss"],
        create_attention_mask=m["create_attention_mask_in_dataloader"],
        allow_ambiguous_pad_tokens=m.get("allow_ambiguous_pad_tokens", False),
        inter_document_masking=True,
        add_bos=bool(m.get("dataloader_add_bos")),
        document_segment_length=m.get("dataloader_document_segment_length"),
    )
    _, valid, _ = BlendedMegatronDatasetBuilder(GPTDataset, [None, None, None], lambda: True, cfg).build()
    return valid[0] if isinstance(valid, list) else valid


def segments(item: dict, seq_len: int) -> list[tuple[int, int]]:
    """Document segments [s, e) of one sample (the dataset pads cu_seqlens with seq_len)."""
    cu = item["cu_seqlens"].tolist()
    ends = [cu[0]] + [c for prev, c in zip(cu, cu[1:]) if c > prev]
    return list(zip(ends[:-1], ends[1:]))


def segment_loss(model, ids: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, chunk: int = 1024) -> float:
    """Summed masked CE of one segment run as a sequence of its own (the LM head in chunks)."""
    dev0 = next(model.parameters()).device
    hidden = model.model(input_ids=ids[None].to(dev0), use_cache=False).last_hidden_state[0]
    head = model.lm_head
    dev = head.weight.device
    hidden, labels, mask = hidden.to(dev), labels.to(dev), mask.to(dev).float()
    total = 0.0
    for i in range(0, hidden.shape[0], chunk):
        ce = torch.nn.functional.cross_entropy(head(hidden[i : i + chunk]).float(), labels[i : i + chunk], reduction="none")
        total += float((ce * mask[i : i + chunk]).sum())
    return total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--config", type=Path, required=True, help="the run's resolved config-<job>.yaml")
    p.add_argument("--prefix", required=True, help="the fixed validation set (indexed dataset prefix)")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--cache", type=Path, help="dataset index cache (default: <out>.cache)")
    p.add_argument("--gpu-mem-gib", type=float, default=None)
    p.add_argument("--engram-tables-cpu", action="store_true")
    p.add_argument("--data-only", action="store_true", help="build the samples and report them; no model")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    m = megatron_args(a.config)
    ds = build_dataset(m, a.prefix, a.seq_len, a.cache or Path(str(a.out) + ".cache"))
    n = len(ds)
    items = [ds[i] for i in range(n)]
    segs = [segments(it, a.seq_len) for it in items]
    tokens = int(sum(float(it["loss_mask"].sum()) for it in items))
    print(f"{a.prefix}: {n} samples of {a.seq_len}, {sum(map(len, segs))} segments, {tokens} loss tokens", flush=True)
    res = {"prefix": a.prefix, "samples": n, "seq_len": a.seq_len, "loss_tokens": tokens,
           "segments": sum(map(len, segs)), "config": str(a.config)}
    if not a.data_only:
        from transformers import AutoConfig, AutoModelForCausalLM

        from .score_on_flag import layer_device_map, load_engram_tables_cpu

        cfg = AutoConfig.from_pretrained(a.model_dir, trust_remote_code=True)
        dmap = layer_device_map(cfg, a.gpu_mem_gib, a.engram_tables_cpu) if a.gpu_mem_gib else "auto"
        t0 = time.time()
        if a.engram_tables_cpu:
            model = load_engram_tables_cpu(a.model_dir, dmap).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(a.model_dir, trust_remote_code=True, dtype=torch.bfloat16,
                                                         device_map=dmap).eval()
        print(f"loaded {a.model_dir} in {time.time() - t0:.0f}s", flush=True)
        per_sample = []
        t0 = time.time()
        with torch.no_grad():
            for k, (it, sg) in enumerate(zip(items, segs)):
                tot = sum(segment_loss(model, it["tokens"][s:e], it["labels"][s:e], it["loss_mask"][s:e]) for s, e in sg)
                per_sample.append(tot)
                if k % 8 == 0:
                    print(f"sample {k}/{n}: {tot / float(it['loss_mask'].sum()):.4f} ({time.time() - t0:.0f}s)", flush=True)
        res.update(model=str(a.model_dir), lm_loss=float(np.sum(per_sample)) / tokens,
                   per_sample_loss_sum=per_sample, nonfinite_samples=[i for i, x in enumerate(per_sample) if not np.isfinite(x)])
    print(json.dumps({k: v for k, v in res.items() if k != "per_sample_loss_sum"}, indent=2))
    a.out.write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
