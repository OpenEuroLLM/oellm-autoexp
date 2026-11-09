#!/usr/bin/env python3
"""Minimal lighteval CLI for MN5: only the accelerate backend, without Typer.

Typer 0.27.3 on MN5 drops the second positional argument of lighteval's
``accelerate`` command. This wrapper resolves the command's OptionInfo defaults
and calls the function directly, preserving the same two positionals:

    python lighteval_accelerate_cli.py accelerate "model_name=...,batch_size=32" \
        "flores200:eng_Latn-deu_Latn|4" --load-tasks-multilingual --output-dir ...
"""

import inspect
import os
import sys

import typer

from lighteval.main_accelerate import accelerate


def _debug_environment() -> None:
    import importlib.util

    from lighteval.utils.imports import is_multilingual_package_available

    for name in ("spacy", "stanza", "jieba", "pyvi"):
        try:
            spec = importlib.util.find_spec(name)
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            spec = f"{type(exc).__name__}: {exc}"
        print(f"[lighteval-debug] find_spec({name}) = {spec}")
    print(
        "[lighteval-debug] is_multilingual_package_available(None) =",
        is_multilingual_package_available(None),
    )
    print("[lighteval-debug] sys.path[:8] =", sys.path[:8])


def _defaults() -> dict:
    out = {}
    for name, param in inspect.signature(accelerate).parameters.items():
        default = param.default
        if isinstance(default, typer.models.OptionInfo):
            out[name] = default.default
        elif default is not inspect.Parameter.empty:
            out[name] = default
    return out


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "accelerate":
        args = args[1:]
    if len(args) < 2 or args[0] in ("--help", "-h"):
        print(
            "usage: lighteval_accelerate_cli.py accelerate <model_args> <tasks> "
            "[--load-tasks-multilingual] [--output-dir DIR] [--save-details] [--max-samples N]"
        )
        return 0
    model_args, tasks = args[0], args[1]
    _debug_environment()
    kwargs = _defaults()
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
            print(
                "usage: lighteval_accelerate_cli.py accelerate <model_args> <tasks> "
                "[--load-tasks-multilingual] [--output-dir DIR] [--save-details] [--max-samples N]"
            )
            return 0
        else:
            i += 1
    accelerate(model_args, tasks, **kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
