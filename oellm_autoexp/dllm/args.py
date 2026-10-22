"""Block-diffusion launcher flags, the chained MegaDLMs extra-args provider,
and validation.

Only argparse is imported at module level, so this file loads without torch. The MegaDLMs
provider is imported inside ``extra_args_provider``, and the block-schedule helpers are imported
inside ``validate_bd_args``. The rules follow the dllm-bd3-spec skill, sections 8, 10, and 12.
"""

from __future__ import annotations

import argparse
import math
import types

# Defaults of the nld flags (spec section 15). Other objectives must leave them at these values.
NLD_DEFAULTS = {
    "bd_nld_ar_weight": 1.0,
    "bd_nld_diffusion_weight": 0.3,
    "bd_nld_ar_only_fraction": 0.2,
}


def add_bd_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the block-diffusion flags to an argument parser.

    Args:
        parser: The parser to extend. It is modified in place.

    Returns:
        The same parser, with a "block diffusion" argument group added.
    """
    group = parser.add_argument_group("block diffusion")
    group.add_argument(
        "--bd-objective",
        type=str,
        choices=["bd3", "ar", "nld"],
        default="bd3",
        help="training objective: bd3 block diffusion, ar causal next-token prediction, or nld "
        "(AR loss on a token-causal clean stream plus weighted block diffusion, spec section 15)",
    )
    group.add_argument(
        "--bd-block-size",
        type=int,
        default=8,
        help="diffusion block size B; (seq_length - 1) must be divisible by B",
    )
    group.add_argument(
        "--bd-block-schedule",
        type=str,
        default="",
        help='block size schedule "frac:B,frac:B"; empty or "none" means a constant --bd-block-size',
    )
    group.add_argument(
        "--bd-token-shift",
        type=int,
        choices=[0, 1],
        default=0,
        help="prediction shift: 0 predicts x0[i] from position i, 1 from position i-1",
    )
    group.add_argument(
        "--bd-complementary-mask",
        type=int,
        choices=[0, 1],
        default=0,
        help="1 adds a complementary masked copy per sequence (bd3 only, block sizes >= 2)",
    )
    group.add_argument(
        "--bd-loss-weighting",
        type=str,
        choices=["elbo", "uniform"],
        default="elbo",
        help="elbo weights each token loss by 1/p of its block, uniform does not",
    )
    group.add_argument(
        "--bd-t-min",
        type=float,
        default=0.0,
        help="lower bound of the per-block masking rate t, drawn from Uniform[t_min, t_max]",
    )
    group.add_argument(
        "--bd-t-max",
        type=float,
        default=1.0,
        help="upper bound of the per-block masking rate t, drawn from Uniform[t_min, t_max]",
    )
    group.add_argument(
        "--bd-doc-mask",
        type=int,
        choices=[0, 1],
        default=1,
        help="1 restricts attention and prediction sources to the same document",
    )
    group.add_argument(
        "--bd-val-seed",
        type=int,
        default=1234,
        help="seed for the deterministic validation masks (monitoring only)",
    )
    group.add_argument(
        "--bd-nld-ar-weight",
        type=float,
        default=NLD_DEFAULTS["bd_nld_ar_weight"],
        help="nld only: weight of the AR cross entropy of the clean stream",
    )
    group.add_argument(
        "--bd-nld-diffusion-weight",
        type=float,
        default=NLD_DEFAULTS["bd_nld_diffusion_weight"],
        help="nld only: weight of the block-diffusion loss of the noisy stream",
    )
    group.add_argument(
        "--bd-nld-ar-only-fraction",
        type=float,
        default=NLD_DEFAULTS["bd_nld_ar_only_fraction"],
        help="nld only: fraction of the iterations that train the plain ar objective first",
    )
    return parser


def extra_args_provider(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the MegaDLMs difflm arguments, then the block-diffusion arguments.

    The launcher passes this function to Megatron as ``extra_args_provider``. The MegaDLMs module
    is imported here, so ``submodules/MegaDLMs`` must be on ``sys.path`` when this runs.

    Args:
        parser: The Megatron argument parser.

    Returns:
        The same parser, with the MegaDLMs and block-diffusion arguments added.
    """
    from custom_args.difflm import extra_args_provider as difflm_extra_args_provider

    parser = difflm_extra_args_provider(parser)
    return add_bd_args(parser)


def validate_bd_args(args: argparse.Namespace | types.SimpleNamespace) -> None:
    """Check the block-diffusion settings against the v1 rules.

    The checks cover parallelism (pipeline 1, context 1, no sequence parallelism), the masking
    rate range, the reset flags, the mask token against the padded vocabulary, the block sizes
    against the effective length L = seq_length - 1, and the complementary-mask requirements.

    Args:
        args: Parsed Megatron arguments, or an object with the same attributes.

    Raises:
        ValueError: If a setting is unsupported or inconsistent with the block-diffusion rules.
    """
    if getattr(args, "pipeline_model_parallel_size", 1) != 1:
        raise ValueError("block diffusion requires pipeline_model_parallel_size == 1")
    if getattr(args, "context_parallel_size", 1) != 1:
        raise ValueError("block diffusion requires context_parallel_size == 1")
    if getattr(args, "sequence_parallel", False):
        raise ValueError("block diffusion v1 does not support sequence parallelism")
    for flag in ("reset_position_ids", "reset_attention_mask"):
        if getattr(args, flag, False):
            raise ValueError(f"block diffusion does not support --{flag.replace('_', '-')}")
    if not 0.0 <= args.bd_t_min <= args.bd_t_max <= 1.0:
        raise ValueError(
            "block diffusion needs 0 <= bd_t_min <= bd_t_max <= 1, "
            f"got bd_t_min={args.bd_t_min} and bd_t_max={args.bd_t_max}"
        )
    if args.mask_token < 0:
        raise ValueError(f"--mask-token must be >= 0, got {args.mask_token}")
    padded_vocab_size = getattr(args, "padded_vocab_size", None)
    if padded_vocab_size is not None and args.mask_token >= padded_vocab_size:
        raise ValueError(
            f"--mask-token {args.mask_token} must be below the padded vocabulary size "
            f"{padded_vocab_size}"
        )

    # Imported here so that this module does not depend on the schedule module at import time.
    from oellm_autoexp.dllm.schedule import parse_block_schedule, validate_block_sizes

    schedule = parse_block_schedule(args.bd_block_schedule)
    validate_block_sizes(args.seq_length - 1, args.bd_block_size, schedule)
    nld = {name: getattr(args, name, default) for name, default in NLD_DEFAULTS.items()}
    if args.bd_objective == "nld":
        ar_weight = nld["bd_nld_ar_weight"]
        diffusion_weight = nld["bd_nld_diffusion_weight"]
        finite = math.isfinite(ar_weight) and math.isfinite(diffusion_weight)
        if not finite or ar_weight < 0 or diffusion_weight < 0 or ar_weight + diffusion_weight == 0:
            raise ValueError(
                "nld needs finite non-negative loss weights that are not both 0, got "
                f"ar {ar_weight} and diffusion {diffusion_weight}"
            )
        fraction = nld["bd_nld_ar_only_fraction"]
        if not 0.0 <= fraction < 1.0:
            raise ValueError(f"--bd-nld-ar-only-fraction must be in [0, 1), got {fraction}")
    else:
        changed = [
            "--" + name.replace("_", "-")
            for name, default in NLD_DEFAULTS.items()
            if nld[name] != default
        ]
        if changed:
            raise ValueError(f"{changed} apply to --bd-objective nld only")
    if args.bd_complementary_mask == 1:
        if args.bd_objective not in ("bd3", "nld"):
            raise ValueError("--bd-complementary-mask 1 requires --bd-objective bd3 or nld")
        block_sizes = [args.bd_block_size, *(size for _, size in schedule)]
        if min(block_sizes) < 2:
            raise ValueError(
                f"--bd-complementary-mask 1 requires block sizes >= 2, got {block_sizes}"
            )
