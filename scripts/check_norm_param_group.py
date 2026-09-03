#!/usr/bin/env python3
"""Check a norm gain's actual Megatron optimizer parameter group.

Run this in place of ``submodules/Megatron-LM/pretrain_gpt.py`` with the same
torchrun/Slurm arguments.  The wrapper builds the real model and optimizer,
reports the matched parameter group's weight-decay settings, and exits before
the first training iteration.

By default it targets the fused attention input RMSNorm in global layer 0 and
requires effective weight decay to be zero.  Environment variables:

* ``NORM_WD_CHECK_PATTERN``: regular expression matched with ``re.search``.
* ``NORM_WD_EXPECTED``: expected effective weight decay (default: ``0``).
* ``NORM_WD_TOLERANCE``: absolute comparison tolerance (default: ``1e-15``).
* ``NORM_WD_CHECK_CONTINUE``: set to ``1`` to continue training after success.
* ``NORM_WD_CHECK_ENTRYPOINT``: alternate training entrypoint.

Example autoexp override::

    backend.launcher_script=./scripts/check_norm_param_group.py

The normal Megatron arguments do not change.
"""

from __future__ import annotations

import json
import os
import re
import runpy
import sys
from pathlib import Path
from typing import Any


DEFAULT_PATTERN = (
    r"(?:^|\.)decoder\.layers\.0\.self_attention\.linear_qkv\.layer_norm_weight$"
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parameter_names(model_chunks: list[Any]) -> dict[int, str]:
    names: dict[int, str] = {}
    for chunk_index, model_chunk in enumerate(model_chunks):
        for name, parameter in model_chunk.named_parameters():
            # A parameter can be exposed through more than one module path. Keep
            # the first path, which is also what Megatron's grouping code sees.
            names.setdefault(id(parameter), f"chunk{chunk_index}.{name}")
    return names


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    megatron_root = repo_root / "submodules" / "Megatron-LM"
    entrypoint = Path(
        os.environ.get("NORM_WD_CHECK_ENTRYPOINT", megatron_root / "pretrain_gpt.py")
    ).resolve()

    if not entrypoint.is_file():
        raise SystemExit(f"Megatron entrypoint does not exist: {entrypoint}")

    sys.path.insert(0, str(megatron_root))

    # Imports intentionally live here so the file can still be compiled or
    # imported on login machines without the training environment's PyTorch.
    import torch
    import torch.distributed as dist
    import megatron.core.optimizer as optimizer_module
    import megatron.training.training as training_module

    pattern_text = os.environ.get("NORM_WD_CHECK_PATTERN", DEFAULT_PATTERN)
    pattern = re.compile(pattern_text)
    expected = float(os.environ.get("NORM_WD_EXPECTED", "0"))
    tolerance = float(os.environ.get("NORM_WD_TOLERANCE", "1e-15"))
    continue_training = _env_flag("NORM_WD_CHECK_CONTINUE")

    original_get_param_groups = optimizer_module._get_param_groups
    original_get_optimizer = training_module.get_megatron_optimizer
    local_matches: list[dict[str, Any]] = []

    def checked_get_param_groups(
        model_chunks: list[Any], config: Any, config_overrides: Any
    ) -> list[dict[str, Any]]:
        groups = original_get_param_groups(model_chunks, config, config_overrides)
        names = _parameter_names(model_chunks)

        for group_index, group in enumerate(groups):
            wd_mult = float(group.get("wd_mult", 1.0))
            base_weight_decay = float(config.weight_decay)
            effective_weight_decay = base_weight_decay * wd_mult
            for parameter in group["params"]:
                name = names.get(id(parameter))
                if name is not None and pattern.search(name):
                    local_matches.append(
                        {
                            "rank": dist.get_rank(),
                            "parameter": name,
                            "shape": list(parameter.shape),
                            "group_index": group_index,
                            "base_weight_decay": base_weight_decay,
                            "wd_mult": wd_mult,
                            "effective_weight_decay": effective_weight_decay,
                            "lr_mult": float(group.get("lr_mult", 1.0)),
                            "max_lr": float(group.get("max_lr", config.lr)),
                        }
                    )
        return groups

    def checked_get_optimizer(*args: Any, **kwargs: Any) -> Any:
        optimizer = original_get_optimizer(*args, **kwargs)

        local_found = len(local_matches)
        local_bad = sum(
            abs(match["effective_weight_decay"] - expected) > tolerance
            for match in local_matches
        )
        collective_device = (
            torch.device("cuda", torch.cuda.current_device())
            if dist.get_backend() == "nccl"
            else torch.device("cpu")
        )
        counts = torch.tensor(
            [local_found, local_bad],
            dtype=torch.int64,
            device=collective_device,
        )
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        global_found, global_bad = (int(value) for value in counts.cpu().tolist())

        rank = dist.get_rank()
        should_report = local_bad > 0 if global_bad else local_found > 0
        reporter = torch.tensor(
            rank if should_report else dist.get_world_size(),
            dtype=torch.int64,
            device=collective_device,
        )
        dist.all_reduce(reporter, op=dist.ReduceOp.MIN)
        if rank == int(reporter.cpu().item()):
            for match in local_matches:
                print("NORM_WD_CHECK " + json.dumps(match, sort_keys=True), flush=True)

        if rank == 0:
            print(
                "NORM_WD_CHECK summary "
                + json.dumps(
                    {
                        "matches": global_found,
                        "mismatches": global_bad,
                        "expected_effective_weight_decay": expected,
                        "pattern": pattern_text,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if global_found == 0:
            raise RuntimeError(
                "Norm weight-decay check found no matching parameter on any rank; "
                f"pattern={pattern_text!r}"
            )
        if global_bad:
            raise RuntimeError(
                f"Norm weight-decay check failed: {global_bad}/{global_found} "
                f"matches differ from expected effective weight decay {expected}"
            )

        if not continue_training:
            dist.barrier()
            if rank == 0:
                print(
                    "NORM_WD_CHECK passed; exiting before the first training iteration",
                    flush=True,
                )
            raise SystemExit(0)

        return optimizer

    optimizer_module._get_param_groups = checked_get_param_groups
    training_module.get_megatron_optimizer = checked_get_optimizer

    # Preserve the original Megatron CLI exactly; runpy executes its normal
    # __main__ path after the two optimizer hooks above are installed.
    sys.argv[0] = str(entrypoint)
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
