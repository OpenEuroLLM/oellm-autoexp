"""GPU parity check of a converted MegaDLMs checkpoint against Hugging Face
Qwen3 (spec 13, I3).

Run with the block-diffusion launcher's Megatron arguments under torchrun, plus ``--load`` of the
converted checkpoint and the ``--parity-*`` flags:

    PYTHONPATH=.:submodules/MegaDLMs torchrun --nproc-per-node <TP> \\
        oellm_autoexp/dllm/eval/parity_check.py <megatron args> --load <converted> --finetune \\
        --no-load-optim --no-load-rng --parity-hf Qwen/Qwen3-0.6B-Base \\
        --parity-data-prefix <held-out prefix> --parity-out parity.json

The model runs in AR mode (``ar_logits``) on ``--parity-num-seqs`` sequences of
``--parity-seq-len`` tokens from the held-out split. Rank 0 compares the logits with
``Qwen3ForCausalLM`` in bfloat16 over the ids below ``len(tokenizer)``: PASS when the top-1
agreement is at least 99 percent and the mean KL(HF || Megatron) is at most 1e-3 nats. I3: with
block size 1, token shift 1 and every position masked, the bd3 per-token cross entropy must equal
the AR next-token cross entropy (within ``--parity-i3-tol`` nats on average).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    for _path in (os.path.join(_REPO_ROOT, "submodules", "MegaDLMs"), _REPO_ROOT):
        if _path not in sys.path:
            sys.path.insert(0, _path)

import torch
import torch.nn.functional as F
from torch import Tensor

TOP1_MIN = 0.99
KL_MAX = 1e-3


def compare_logits(reference: Tensor, candidate: Tensor, real_vocab: int) -> dict[str, float]:
    """Compare two logit tensors over the real vocabulary.

    Args:
        reference: Reference logits [b, s, >= real_vocab] (Hugging Face).
        candidate: Candidate logits [b, s, >= real_vocab] (Megatron).
        real_vocab: Number of real token ids (``len(tokenizer)``); padding rows and the mask
            token are excluded.

    Returns:
        ``top1`` (fraction of positions with the same argmax), ``kl_mean`` and ``kl_max``
        (KL(reference || candidate) in nats per position), and ``positions``.
    """
    ref = reference[..., :real_vocab].float()
    cand = candidate[..., :real_vocab].float()
    if ref.shape != cand.shape:
        raise ValueError(f"shape mismatch {tuple(ref.shape)} vs {tuple(cand.shape)}")
    top1 = (ref.argmax(-1) == cand.argmax(-1)).float().mean().item()
    log_p = F.log_softmax(ref, dim=-1)
    log_q = F.log_softmax(cand, dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1)
    return {
        "top1": top1,
        "kl_mean": kl.mean().item(),
        "kl_max": kl.max().item(),
        "positions": int(kl.numel()),
    }


def i3_compare(bd3_logits: Tensor, ar_logits: Tensor, x0: Tensor) -> dict[str, float]:
    """Compare bd3 (B = 1, shift 1, all masked) and AR per-token cross entropy
    (invariant I3).

    ``bd3_logits[:, i]`` predicts ``x0[:, i]`` (source = clean position ``i - 1``) and
    ``ar_logits[:, i - 1]`` predicts ``x0[:, i]``; positions ``1..L-1`` are compared.

    Args:
        bd3_logits: [b, L, V] from ``bd3_logits(x0, xt=all mask, block_size=1, token_shift=1)``.
        ar_logits: [b, L, V] from ``ar_logits(x0)``.
        x0: Clean tokens [b, L].

    Returns:
        ``mean_abs_diff`` and ``max_abs_diff`` of the per-token cross entropy in nats, and the
        mean AR cross entropy ``ar_ce_mean``.
    """
    target = x0[:, 1:]
    ce_bd3 = F.cross_entropy(bd3_logits[:, 1:].float().transpose(1, 2), target, reduction="none")
    ce_ar = F.cross_entropy(ar_logits[:, :-1].float().transpose(1, 2), target, reduction="none")
    diff = (ce_bd3 - ce_ar).abs()
    return {
        "mean_abs_diff": diff.mean().item(),
        "max_abs_diff": diff.max().item(),
        "ar_ce_mean": ce_ar.mean().item(),
    }


def parity_verdict(logit_stats: dict[str, float], i3_stats: dict[str, float], i3_tol: float):
    """Return (parity_pass, i3_pass) for the thresholds of spec section 13.

    Args:
        logit_stats: Output of ``compare_logits``.
        i3_stats: Output of ``i3_compare``.
        i3_tol: Allowed mean absolute cross-entropy difference for I3, in nats.

    Returns:
        A tuple of two booleans.
    """
    parity = logit_stats["top1"] >= TOP1_MIN and logit_stats["kl_mean"] <= KL_MAX
    i3 = math.isfinite(i3_stats["mean_abs_diff"]) and i3_stats["mean_abs_diff"] <= i3_tol
    return parity, i3


def add_parity_args(parser):
    """Add the ``--parity-*`` flags to a Megatron argument parser.

    Args:
        parser: The argument parser.

    Returns:
        The same parser.
    """
    group = parser.add_argument_group(title="parity check")
    group.add_argument(
        "--parity-hf",
        type=str,
        required=False,
        default=None,
        help="HF model id or directory of the source Qwen3 checkpoint",
    )
    group.add_argument(
        "--parity-data-prefix",
        type=str,
        default=None,
        help="Megatron .bin/.idx prefix of the held-out split",
    )
    group.add_argument("--parity-num-seqs", type=int, default=4, help="number of sequences")
    group.add_argument("--parity-seq-len", type=int, default=512, help="tokens per sequence")
    group.add_argument(
        "--parity-i3-tol",
        type=float,
        default=1e-2,
        help="mean absolute CE difference allowed for I3 (nats, bf16)",
    )
    group.add_argument("--parity-out", type=str, default=None, help="JSON result path")
    return parser


def _parity_extra_args_provider(parser):
    from oellm_autoexp.dllm.args import extra_args_provider

    return add_parity_args(extra_args_provider(parser))


def _load_windows(prefix: str, num: int, seq_len: int) -> Tensor:
    import numpy as np

    from oellm_autoexp.dllm.data_prep import read_idx_header

    header = read_idx_header(Path(prefix + ".idx"))
    tokens = np.memmap(prefix + ".bin", dtype=header["dtype"], mode="r")
    if tokens.shape[0] < num * seq_len:
        raise ValueError(f"{prefix} has {tokens.shape[0]} tokens, need {num * seq_len}")
    window = np.asarray(tokens[: num * seq_len], dtype=np.int64).reshape(num, seq_len)
    return torch.from_numpy(window)


def main() -> int:
    """Run the parity check; returns 0 when parity and I3 pass on rank 0."""
    from megatron.core.enums import ModelType
    from megatron.training import get_args, get_model, get_tokenizer, initialize_megatron
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.utils import unwrap_model

    from oellm_autoexp.dllm.pretrain_block_difflm import model_provider

    initialize_megatron(
        extra_args_provider=_parity_extra_args_provider,
        args_defaults={"tokenizer_type": "HuggingFaceTokenizer"},
    )
    args = get_args()
    if args.parity_hf is None or args.parity_data_prefix is None:
        raise ValueError("--parity-hf and --parity-data-prefix are required")
    model = get_model(model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)
    load_checkpoint(model, None, None)
    net = unwrap_model(model[0])
    net.eval()
    real_vocab = len(get_tokenizer()._tokenizer)
    x0 = _load_windows(args.parity_data_prefix, args.parity_num_seqs, args.parity_seq_len).cuda()
    xt = torch.full_like(x0, int(args.mask_token))
    with torch.no_grad():
        meg_ar = net.ar_logits(x0)
        meg_bd3 = net.bd3_logits(x0, xt, block_size=1, token_shift=1)

    rank = torch.distributed.get_rank()
    passed = True
    if rank == 0:
        from transformers import AutoModelForCausalLM

        hf = AutoModelForCausalLM.from_pretrained(
            args.parity_hf, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).cuda()
        hf.eval()
        with torch.no_grad():
            hf_logits = hf(input_ids=x0).logits
        logit_stats = compare_logits(hf_logits, meg_ar, real_vocab)
        i3_stats = i3_compare(meg_bd3[..., :real_vocab], meg_ar[..., :real_vocab], x0)
        parity_ok, i3_ok = parity_verdict(logit_stats, i3_stats, args.parity_i3_tol)
        result = {
            "load": args.load,
            "hf": args.parity_hf,
            "tensor_model_parallel_size": args.tensor_model_parallel_size,
            "num_seqs": args.parity_num_seqs,
            "seq_len": args.parity_seq_len,
            "real_vocab": real_vocab,
            "logits": logit_stats,
            "i3": i3_stats,
            "parity_pass": parity_ok,
            "i3_pass": i3_ok,
        }
        print(json.dumps(result), flush=True)
        print(
            f"PARITY {'PASS' if parity_ok else 'FAIL'}: top1={logit_stats['top1']:.4f} "
            f"kl_mean={logit_stats['kl_mean']:.2e}",
            flush=True,
        )
        print(
            f"I3 {'PASS' if i3_ok else 'FAIL'}: mean_abs_diff={i3_stats['mean_abs_diff']:.2e}",
            flush=True,
        )
        if args.parity_out:
            Path(args.parity_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.parity_out).write_text(json.dumps(result, indent=2))
        passed = parity_ok and i3_ok
    flag = torch.tensor([1 if passed else 0], device="cuda")
    torch.distributed.broadcast(flag, src=0)
    torch.distributed.barrier()
    return 0 if bool(flag.item()) else 1


if __name__ == "__main__":
    sys.exit(main())
