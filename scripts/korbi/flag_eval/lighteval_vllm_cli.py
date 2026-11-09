#!/usr/bin/env python3
"""Minimal lighteval CLI for MN5: vLLM backend, manual argument parsing.

Typer 0.27.3 on MN5 drops the second positional argument of lighteval's
commands, so this wrapper resolves the command's OptionInfo defaults and calls
``lighteval.main_vllm.vllm`` directly, mirroring ``lighteval_accelerate_cli.py``.

The vLLM backend is used because the 248 GB hybrid export does not fit the HF
``accelerate`` path (CPU offload then CUDA-OOMs); vLLM is the same engine the
lm-eval and evalchemy rows already use.

    python lighteval_vllm_cli.py vllm "model_name=...,dtype=bfloat16,..." \
        "flores200:eng_Latn-swe_Latn|4" --load-tasks-multilingual --output-dir ...
"""

import inspect
import sys

import typer

from lighteval.main_vllm import vllm


def _defaults() -> dict:
    out = {}
    for name, param in inspect.signature(vllm).parameters.items():
        default = param.default
        if isinstance(default, typer.models.OptionInfo):
            out[name] = default.default
        elif default is not inspect.Parameter.empty:
            out[name] = default
    return out


def _usage() -> None:
    print(
        "usage: lighteval_vllm_cli.py vllm <model_args> <tasks> "
        "[--load-tasks-multilingual] [--output-dir DIR] [--save-details] [--max-samples N]"
    )


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "vllm":
        args = args[1:]
    if len(args) < 2 or args[0] in ("--help", "-h"):
        _usage()
        return 0
    model_args, tasks = args[0], args[1]
    kwargs = _defaults()
    # Required positionals are passed explicitly, never through the defaults.
    kwargs.pop("model_args", None)
    kwargs.pop("tasks", None)
    i = 2
    while i < len(args):
        arg = args[i]
        if arg == "--load-tasks-multilingual":
            kwargs["load_tasks_multilingual"] = True
            i += 1
        elif arg == "--output-dir":
            kwargs["output_dir"] = args[i + 1]
            i += 2
        elif arg == "--save-details":
            kwargs["save_details"] = True
            i += 1
        elif arg == "--max-samples":
            kwargs["max_samples"] = int(args[i + 1])
            i += 2
        elif arg in ("--help", "-h"):
            _usage()
            return 0
        else:
            i += 1
    vllm(model_args, tasks, **kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
