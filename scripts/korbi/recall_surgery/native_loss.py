#!/usr/bin/env python3
"""The NATIVE Megatron model's loss on a fixed token batch (the paired-loss check of the recall-surgery HF export,
scripts/korbi/recall_surgery/ladder17b_paired_check.sh). Run by ladder17b_train.sh (ENTRY=this file, NPROC=1) so it
gets exactly the training CLI (arm spec, geometry, SKDA_INTERVENTION code dir) plus --skip-train --no-load-optim
--no-load-rng --finetune: no optimizer, no DDP wrapper, the checkpoint's weights only.

The batch: RECALL_TOKENS (an .npy of shape [n, seq+1], int64) if it exists, else the first n x (seq+1) tokens of the
validation prefix's documents concatenated (written to RECALL_TOKENS for the HF side). Writes RECALL_NATIVE_OUT
(json: per-sequence mean loss and the overall mean).
"""
import json
import os
from functools import partial

import numpy as np
import torch


def fixed_tokens(prefix: str, n: int, seq: int) -> np.ndarray:
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    ds = IndexedDataset(prefix)
    need, parts, i = n * (seq + 1), [], 0
    while sum(len(p) for p in parts) < need:
        parts.append(np.asarray(ds[i], dtype=np.int64))
        i += 1
    return np.concatenate(parts)[:need].reshape(n, seq + 1)


def main():
    from megatron.core.enums import ModelType
    from megatron.training import get_args
    from megatron.training.arguments import parse_and_validate_args
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.initialize import initialize_megatron
    from megatron.training.training import get_model

    from gpt_builders import gpt_builder
    from model_provider import model_provider

    parse_and_validate_args(args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
    initialize_megatron()
    args = get_args()
    model = get_model(partial(model_provider, gpt_builder), ModelType.encoder_or_decoder, wrap_with_ddp=False)
    load_checkpoint(model, None, None)
    m = model[0]
    m.eval()

    tok_path = os.environ['RECALL_TOKENS']
    n, seq = int(os.environ.get('RECALL_N', '8')), int(args.seq_length)
    if os.path.exists(tok_path):
        tokens = np.load(tok_path)
    else:
        tokens = fixed_tokens(args.valid_data_path[0], n, seq)
        np.save(tok_path, tokens)
    x = torch.from_numpy(tokens).cuda()
    losses = []
    with torch.no_grad():
        for row in x:
            inp, lab = row[None, :-1], row[None, 1:]
            pos = torch.arange(inp.shape[1], device=inp.device)[None]
            out = m(inp, pos, None, labels=lab)  # per-token loss [1, seq]
            losses.append(float(out.float().mean()))
    res = {'per_sequence': losses, 'mean': float(np.mean(losses)), 'n': len(losses), 'seq': seq,
           'intervention': os.environ.get('SKDA_INTERVENTION', ''), 'load': args.load}
    print('[native_loss]', json.dumps(res), flush=True)
    with open(os.environ['RECALL_NATIVE_OUT'], 'w') as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
