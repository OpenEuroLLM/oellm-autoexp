#!/usr/bin/env python3
"""Megatron half of the HF<->Megatron parity check.

Loads a training run's ``torch_dist`` checkpoint with the *real* Megatron stack
(same container, same arg list the run itself was launched with) and dumps the
logits for a fixed set of token ids. ``check_hf_megatron_parity.sh`` then feeds
the identical ids through the converted HF model and compares.

The architecture is not re-specified here: the Megatron flags are lifted
verbatim out of the run's own ``job.sbatch``, so the model is built exactly as
it was trained. Only the things that would pull in the data pipeline, the
logger, or the training schedule are dropped or overridden.

Must run **inside the training container** (it imports megatron.core and TE):

    apptainer exec --nv ... python scripts/korbi/dump_megatron_logits.py \\
        --run-dir <train run dir> --tokens tokens.npy --out megatron_logits.pt
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from pathlib import Path

import numpy as np
import torch

# Flags that would drag in the dataset builder / logging / schedule. The model
# architecture is untouched by all of these.
_DROP_WITH_VALUE = {
    "--data-path",
    "--train-data-path",
    "--valid-data-path",
    "--test-data-path",
    "--data-cache-path",
    "--split",
    "--save",
    "--save-interval",
    "--save-extra-steps",
    "--tensorboard-dir",
    "--wandb-project",
    "--wandb-exp-name",
    "--wandb-save-dir",
    "--wandb-entity",
    "--load",
    "--ckpt-step",
    "--num-workers",
    "--non-persistent-save-interval",
    "--non-persistent-ckpt-type",
    "--eval-interval",
    "--eval-iters",
    "--micro-batch-size",
    "--global-batch-size",
    "--cuda-graph-impl",
    "--rerun-mode",
}
_DROP_FLAGS = {
    "--log-progress",
    "--log-throughput",
    "--log-timers-to-tensorboard",
    "--use-distributed-optimizer",
    "--overlap-grad-reduce",
    "--overlap-param-gather",
}


_LAUNCHERS = ("pretrain_hybrid.py", "pretrain_gpt.py")


def find_launcher(text: str) -> str:
    """Which Megatron entry point the run used (HybridModel or GPTModel)."""
    for launcher in _LAUNCHERS:
        if launcher in text:
            return launcher
    raise ValueError(f"no {' / '.join(_LAUNCHERS)} invocation found")


def extract_megatron_args(sbatch: Path) -> list[str]:
    """Pull the ``--flag [value…]`` list that follows the Megatron entry
    point."""
    text = sbatch.read_text()
    launcher = find_launcher(text)
    tail = text[text.find(launcher) + len(launcher) :]
    # The command ends at the first unescaped newline (the block is line-continued).
    tail = tail.replace("\\\n", " ")
    tail = tail.split("\n", 1)[0]
    tail = re.sub(r"'\s*$", "", tail)
    # Values are shell-quoted when they need it (e.g. a pattern with '|').
    tokens = shlex.split(tail)
    return filter_megatron_args(tokens)


def filter_megatron_args(tokens: list[str]) -> list[str]:
    """Drop the data/logging/schedule flags; keep everything that shapes the
    model."""

    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            i += 1
            continue
        if "=" in tok:  # --flag=value form (values that start with "-")
            if tok.split("=", 1)[0] not in _DROP_WITH_VALUE:
                out.append(tok)
            i += 1
            continue
        # Collect this flag's values (everything up to the next --flag).
        j = i + 1
        values = []
        while j < len(tokens) and not tokens[j].startswith("--"):
            values.append(tokens[j])
            j += 1
        if tok in _DROP_WITH_VALUE or tok in _DROP_FLAGS:
            i = j
            continue
        out.append(tok)
        out.extend(values)
        i = j
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--tokens", type=Path, required=True, help=".npy int array [b, s]")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--iter", type=int, default=None)
    p.add_argument(
        "--args-file",
        type=Path,
        default=None,
        help="Megatron flags (one shell-quoted line) instead of <run-dir>/job.sbatch, for "
        "runs not launched by oellm-autoexp; the entry point is named on the line.",
    )
    args, _ = p.parse_known_args()

    tokens = np.load(args.tokens)
    if tokens.ndim != 2:
        raise ValueError(f"{args.tokens}: expected [b, s], got {tokens.shape}")
    batch, seq = tokens.shape

    if args.args_file is not None:
        text = args.args_file.read_text()
        launcher = find_launcher(text)
        megatron_args = filter_megatron_args(shlex.split(text.split(launcher, 1)[1]))
    else:
        launcher = find_launcher((args.run_dir / "job.sbatch").read_text())
        megatron_args = extract_megatron_args(args.run_dir / "job.sbatch")
    megatron_args += [
        "--load",
        str(args.run_dir),
        "--no-load-optim",
        "--no-load-rng",
        "--exit-on-missing-checkpoint",
        "--micro-batch-size",
        "1",
        "--global-batch-size",
        str(batch),
        "--eval-iters",
        "0",
        "--seq-length",
        str(max(seq, 128)),
        "--max-position-embeddings",
        str(max(seq, 4096)),
    ]
    if args.iter is not None:
        megatron_args += ["--ckpt-step", str(args.iter)]

    sys.argv = [launcher] + megatron_args

    # Single-process, single-GPU run.
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    os.environ.setdefault("LOCAL_RANK", "0")

    from functools import partial  # noqa: E402

    from megatron.core.enums import ModelType  # noqa: E402
    from megatron.training.arguments import parse_and_validate_args  # noqa: E402
    from megatron.training.checkpointing import load_checkpoint  # noqa: E402
    from megatron.training.initialize import initialize_megatron  # noqa: E402
    from megatron.training.training import get_model  # noqa: E402

    if launcher == "pretrain_hybrid.py":
        from hybrid_builders import hybrid_builder as builder  # noqa: E402
        from pretrain_hybrid import model_provider as _model_provider  # noqa: E402
    else:
        from gpt_builders import gpt_builder as builder  # noqa: E402
        from pretrain_gpt import model_provider as _model_provider  # noqa: E402

    # This fork parses+validates args first and only then initialises globals
    # (pretrain_gpt.py does the same), so `initialize_megatron` alone would trip
    # over an uninitialised arg store.
    margs = parse_and_validate_args(args_defaults={"tokenizer_type": "GPT2BPETokenizer"})
    initialize_megatron(allow_no_cuda=False)
    model_provider = partial(_model_provider, builder)
    print(
        f"[dump] layers={margs.num_layers} hidden={margs.hidden_size} "
        f"variant={getattr(margs, 'experimental_attention_variant', None)} "
        f"pe={margs.position_embedding_type} window={getattr(margs, 'window_size', None)}",
        flush=True,
    )

    model = get_model(model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)
    loaded_iter = load_checkpoint(model, None, None)
    print(f"[dump] loaded checkpoint iteration: {loaded_iter}", flush=True)

    net = model[0]
    net.eval()

    ids = torch.from_numpy(tokens.astype(np.int64)).cuda()
    position_ids = torch.arange(seq, device=ids.device).unsqueeze(0).expand(batch, seq)
    with torch.no_grad():
        logits = net(ids, position_ids, None)
    logits = logits.float().cpu()
    print(f"[dump] logits {tuple(logits.shape)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input_ids": ids.cpu(), "logits": logits}, args.out)
    print(f"[dump] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
