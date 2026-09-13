#!/usr/bin/env python3
"""Run Megatron's real validate_args() against a rendered sbatch, without a
job.

WHY
---
`validate_args` fires inside `initialize_megatron`, i.e. after SLURM has granted
an allocation, started the container and initialised torch.distributed. A config
that trips one of its ~400 asserts therefore costs a full queue-and-launch cycle
to discover, and the traceback lands in a slurm log rather than in front of you.
That is how `save_interval: null` (arguments.py:863) ate jobs 1492415/16/17.

Nothing in `validate_args` needs a GPU or a process group -- it only reads
`args`, `RANK` and `WORLD_SIZE`. So we can parse the argv straight out of the
rendered sbatch and validate it locally in about a second.

This checks ARGUMENT-LEVEL validity only. It cannot catch anything that happens
later: missing data paths, tokenizer mismatches, OOM, or NCCL trouble.

USAGE
-----
    # every arm of a sweep, straight from the rendered scripts
    python scripts/korbi/preflight_megatron_args.py \
        /e/project1/.../oellm_32b_dense_pda-*/job_*.sbatch

    # simulate the real geometry (defaults to the sbatch's own node count x 4)
    python scripts/korbi/preflight_megatron_args.py --world-size 4 run.sbatch

Exit status is the number of scripts that failed validation.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Megatron imports transformer_engine at module scope; mock it so this runs
# outside the container. Mirrors scripts/generate_megatron_config.py.
for _module in (
    "transformer_engine",
    "transformer_engine.pytorch",
    "transformer_engine.pytorch.router",
    "transformer_engine.pytorch.cpp_extensions",
    "transformer_engine.pytorch.distributed",
    "transformer_engine.pytorch.tensor",
    "transformer_engine.pytorch.float8_tensor",
    "transformer_engine.pytorch.fp8",
):
    sys.modules.setdefault(_module, MagicMock())

# mcore 0.19 reads te.__version__ at import time
# (tensor_parallel/generalized_tensor_parallelism.py:48), and a bare MagicMock raises
# AttributeError for dunders. Pin it to the TE the deployment container ships
# (nemo_26.04 -> 2.14.0) so any version-gated validation matches what will really run.
sys.modules["transformer_engine"].__version__ = "2.14.0"

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _megatron_path_from_sbatch(scripts: list[str]) -> Path:
    """Which Megatron checkout to validate against -- taken from the sbatch
    itself.

    THIS MUST NOT BE HARDCODED. It was, and the moment the backend configs were
    repointed from submodules/Megatron-LM to submodules/Megatron-LM-v0.19 the preflight
    started validating 0.19 configs against the 0.16 parser and rejected a perfectly
    valid `--dataloader-inter-document-masking` as "unrecognized arguments". A preflight
    that checks a different tree than the job will run is worse than none: it produces
    confident false failures.

    The rendered sbatch names the checkout in its PYTHONPATH/RUN_DIR, so read it there
    and fall back to the default submodule only when nothing matches.
    """
    pat = re.compile(r"submodules/(Megatron-LM[A-Za-z0-9._-]*)")
    for script in scripts:
        try:
            text = Path(script).read_text(errors="replace")
        except OSError:
            continue
        found = pat.findall(text)
        if found:
            # Longest match wins so "Megatron-LM-v0.19" is not truncated to "Megatron-LM".
            return _REPO_ROOT / "submodules" / max(found, key=len)
    return _REPO_ROOT / "submodules" / "Megatron-LM"


_MEGATRON_PATH = _megatron_path_from_sbatch([a for a in sys.argv[1:] if not a.startswith("-")])
sys.path.insert(0, str(_MEGATRON_PATH))

NODES_RE = re.compile(r"^#SBATCH\s+--nodes[= ](\d+)", re.MULTILINE)
JOBNAME_RE = re.compile(r"^#SBATCH\s+--job-name[= ](\S+)", re.MULTILINE)
EXPORT_RE = re.compile(r"^export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.MULTILINE)


def sbatch_env(script: Path) -> dict[str, str]:
    """Literal `export NAME=VALUE` pairs from the sbatch.

    validate_args reads the environment, not just argv -- e.g.
    arguments.py:959 rejects TP>1 or CP>1 unless
    CUDA_DEVICE_MAX_CONNECTIONS=1, which every JUPITER experiment sets
    itself because config/slurm/jupiter.yaml does not. Without this the
    preflight false-fails any TP>1 config, or blames the wrong check.

    Values containing shell expansion ($VAR, $(cmd)) are skipped -- they
    cannot be resolved off-cluster, and guessing would be worse than
    leaving them unset.
    """
    env = {}
    for name, value in EXPORT_RE.findall(script.read_text()):
        value = value.strip().strip('"').strip("'")
        if "$" in value:
            continue
        env[name] = value
    return env


def label(script: Path) -> str:
    """Prefer the sbatch's own job name -- run directories are not always
    distinct."""
    match = JOBNAME_RE.search(script.read_text())
    return match.group(1) if match else script.name


def extract_argv(script: Path) -> list[str]:
    """Pull the pretrain_gpt.py argument list out of a rendered sbatch.

    The whole launch is wrapped in `srun ... bash -c '...'`, so the argument list
    sits inside a single-quoted shell string -- shlex would hand back one token.
    Cut at the marker instead and drop the quote that closes the bash -c.
    """
    # Join line continuations so the invocation is one contiguous string.
    flat = script.read_text().replace("\\\n", " ")
    marker = "pretrain_gpt.py"
    if marker not in flat:
        raise ValueError(f"no {marker} invocation found in {script}")
    rest = flat.split(marker, 1)[1]
    # The last argument carries the closing quote of `bash -c '`.
    rest = rest.rsplit("'", 1)[0] if rest.rstrip().endswith("'") else rest
    return shlex.split(rest, comments=True)


def sbatch_world_size(script: Path) -> int:
    """Nodes x 4 GPUs, the JUPITER default."""
    match = NODES_RE.search(script.read_text())
    return int(match.group(1)) * 4 if match else 4


def extra_checks(args) -> str | None:
    """Traps that validate_args does NOT catch, learned the expensive way.

    These fire later than validate_args -- at dataset build or first
    step -- so they still cost an allocation to discover. Each entry
    here is one that actually burned jobs.
    """
    # eval_iters: 0 looks like "skip validation", and the do_valid gate at
    # training.py:2931 seems to agree. But that gate is inside the
    # `if getattr(args, 'perform_rl_step', True)` branch; ordinary pretraining
    # takes the else at :2934 and builds a valid dataloader unconditionally. The
    # valid dataset is then sized to 0 samples and data_samplers.py:133 asserts
    # `no sample to consume: 0`. Cost: jobs 1492477/78/79.
    if args.eval_iters == 0 and not args.skip_train:
        return (
            "eval_iters == 0 does not skip the validation dataloader outside the "
            "RL path; it starves it to 0 samples and data_samplers.py:133 will "
            "assert 'no sample to consume: 0'. Leave eval_iters > 0 and push "
            "eval_interval beyond exit_interval instead."
        )
    return None


def validate(script: Path, world_size: int | None) -> str | None:
    """Return None on success, else the failure message."""
    from megatron.training.arguments import parse_args, validate_args

    argv = extract_argv(script)
    size = world_size or sbatch_world_size(script)

    saved_argv, saved_env = sys.argv, dict(os.environ)
    sys.argv = ["pretrain_gpt.py", *argv]
    # The job's own environment first; WORLD_SIZE/RANK are ours and must win.
    os.environ.update(sbatch_env(script))
    os.environ["WORLD_SIZE"] = str(size)
    os.environ["RANK"] = "0"
    try:
        args = parse_args(ignore_unknown_args=False)
        validate_args(args)
        return extra_checks(args)
    except SystemExit as exc:  # argparse rejected an unknown/malformed flag
        return f"argparse rejected the command line (exit {exc.code})"
    except AssertionError as exc:
        import traceback

        frame = traceback.extract_tb(exc.__traceback__)[-1]
        detail = str(exc) or frame.line
        return f"{Path(frame.filename).name}:{frame.lineno}  {detail}"
    except Exception as exc:  # noqa: BLE001 - report whatever validate_args raised
        return f"{type(exc).__name__}: {exc}"
    finally:
        sys.argv = saved_argv
        os.environ.clear()
        os.environ.update(saved_env)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("scripts", nargs="+", type=Path, help="rendered .sbatch file(s)")
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="override the simulated world size (default: sbatch --nodes x 4)",
    )
    args = parser.parse_args()

    failures = 0
    for script in args.scripts:
        if not script.is_file():
            print(f"FAIL  {script}  (not a file)")
            failures += 1
            continue
        try:
            problem = validate(script, args.world_size)
        except ValueError as exc:
            problem = str(exc)
        if problem:
            print(f"FAIL  {label(script)}\n      {problem}")
            failures += 1
        else:
            print(f"OK    {label(script)}")

    print(f"\n{len(args.scripts) - failures}/{len(args.scripts)} passed validate_args")
    return failures


if __name__ == "__main__":
    sys.exit(main())
