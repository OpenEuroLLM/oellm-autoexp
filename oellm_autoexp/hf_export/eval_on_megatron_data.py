"""Score a converted HF checkpoint on real Megatron-pretokenized data.

The short synthetic probe in ``validate_hf_export.py`` only proves a model is
"not random". This is the check with an actual reference value: run the HF port
over token spans taken straight from the pretokenized training mix and compare
the mean NLL against the ``lm loss value`` the training run itself reported
(``grep 'validation loss at iteration' <run_dir>/current.log``). A correct
conversion lands within a few hundredths of that number; anything materially
above it is a conversion bug.

Reading the data needs no Megatron: the ``.idx`` header carries the dtype and
the ``.bin`` is a flat array of token ids, so a contiguous span of it is exactly
what the packed training/validation batches are built from.

Usage::

    python -m oellm_autoexp.hf_export.eval_on_megatron_data \\
        --model-dir <hf_dir> \\
        --data-prefix /e/data1/.../openeurollm-tokenized-256k/dclm-baseline-1.0-10p-sample/dclm-10p-sample \\
        --seq-len 2048 --num-seqs 8
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

# Megatron IndexedDataset dtype codes (megatron/core/datasets/indexed_dataset.py).
_DTYPES = {
    1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32,
    5: np.int64, 6: np.float64, 7: np.float32, 8: np.uint16,
}
_HDR_MAGIC = b"MMIDIDX\x00\x00"


def read_index_dtype(idx_path: Path) -> np.dtype:
    with idx_path.open("rb") as f:
        magic = f.read(9)
        if magic != _HDR_MAGIC:
            raise ValueError(f"{idx_path}: unexpected magic {magic!r}")
        (_version,) = struct.unpack("<Q", f.read(8))
        (code,) = struct.unpack("<B", f.read(1))
    if code not in _DTYPES:
        raise ValueError(f"{idx_path}: unknown dtype code {code}")
    return np.dtype(_DTYPES[code])


def load_spans(prefix: Path, seq_len: int, num_seqs: int, skip: int) -> np.ndarray:
    dtype = read_index_dtype(prefix.with_suffix(".idx"))
    tokens = np.memmap(prefix.with_suffix(".bin"), dtype=dtype, mode="r")
    need = (num_seqs + 1) * seq_len + skip
    if tokens.shape[0] < need:
        raise ValueError(f"{prefix}.bin has {tokens.shape[0]} tokens, need {need}")
    span = np.asarray(tokens[skip : skip + num_seqs * seq_len], dtype=np.int64)
    return span.reshape(num_seqs, seq_len)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--data-prefix", type=Path, required=True, help="path without .bin/.idx")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--num-seqs", type=int, default=8)
    p.add_argument("--skip", type=int, default=1_000_000, help="token offset into the .bin")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, dtype=torch.bfloat16
    )
    model.to(args.device).eval()

    batches = load_spans(args.data_prefix, args.seq_len, args.num_seqs, args.skip)
    print(f"model                : {args.model_dir.name}")
    print(f"mixer_types          : {cfg.mixer_types[0]} … ({len(set(cfg.mixer_types))} kinds)")
    print(f"position_embedding   : {cfg.position_embedding_type}  rope_theta={cfg.rope_theta}")
    print(f"sliding_window       : {cfg.sliding_window}")
    print(f"data                 : {args.data_prefix.name} {batches.shape}")

    total_nll = 0.0
    total_tok = 0
    for row in batches:
        ids = torch.from_numpy(row).unsqueeze(0).to(args.device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits.float()
        nll = torch.nn.functional.cross_entropy(
            logits[0, :-1], ids[0, 1:], reduction="sum"
        ).item()
        total_nll += nll
        total_tok += ids.shape[1] - 1

    mean = total_nll / total_tok
    print(f"tokens scored        : {total_tok}")
    print(f"MEAN NLL             : {mean:.4f}   (PPL {np.exp(mean):.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
