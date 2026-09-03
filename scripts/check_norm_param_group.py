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
* ``NORM_WD_CHECK_ALLOW_TRAINING_IO``: retain the original save/logging paths
  in check-only mode. Checkpoint loading is always retained so the assertion
  examines the restored optimizer; only outputs are redirected by default.

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


def _remove_cli_option(argv: list[str], option: str, *, takes_value: bool) -> list[str]:
    result = [argv[0]]
    index = 1
    while index < len(argv):
        argument = argv[index]
        if argument == option:
            index += 2 if takes_value else 1
            continue
        if takes_value and argument.startswith(option + "="):
            index += 1
            continue
        result.append(argument)
        index += 1
    return result


def _redirect_training_output(argv: list[str]) -> tuple[list[str], Path]:
    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    scratch = Path("/tmp") / f"oellm-norm-wd-check-{job_id}"

    for option in (
        "--save",
        "--non-persistent-global-ckpt-dir",
        "--non-persistent-local-ckpt-dir",
        "--tensorboard-dir",
        "--wandb-save-dir",
        "--wandb-project",
        "--wandb-exp-name",
        "--wandb-entity",
    ):
        argv = _remove_cli_option(argv, option, takes_value=True)

    for option in ("--log-progress", "--async-save", "--use-persistent-ckpt-worker"):
        argv = _remove_cli_option(argv, option, takes_value=False)

    # No checkpoint should be written because check-only mode exits directly
    # after optimizer construction. Redirect these paths as a second guard.
    argv.extend(
        [
            "--save",
            str(scratch / "checkpoints"),
            "--non-persistent-global-ckpt-dir",
            str(scratch / "checkpoints_rolling"),
            "--tensorboard-dir",
            str(scratch / "tensorboard"),
        ]
    )
    return argv, scratch


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

    if not continue_training and not _env_flag("NORM_WD_CHECK_ALLOW_TRAINING_IO"):
        sys.argv, scratch = _redirect_training_output(sys.argv)
        if os.environ.get("RANK", "0") == "0":
            print(
                "NORM_WD_CHECK retained checkpoint loading and redirected "
                f"save/log paths to {scratch}",
                flush=True,
            )

    original_get_param_groups = optimizer_module._get_param_groups
    original_setup_model_and_optimizer = training_module.setup_model_and_optimizer
    captured_matches: list[dict[str, Any]] = []

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
                    captured_matches.append(
                        {
                            "parameter": name,
                            "parameter_object": parameter,
                            "shape": list(parameter.shape),
                            "group_index": group_index,
                            "preload_wd_mult": wd_mult,
                            "preload_effective_weight_decay": effective_weight_decay,
                        }
                    )
        return groups

    def leaf_optimizers(optimizer: Any) -> list[Any]:
        chained = getattr(optimizer, "chained_optimizers", None)
        if chained is None:
            return [optimizer]
        leaves: list[Any] = []
        for child in chained:
            leaves.extend(leaf_optimizers(child))
        return leaves

    def find_runtime_group(
        optimizer: Any, parameter: Any, fallback_index: int
    ) -> tuple[Any, Any]:
        leaves = leaf_optimizers(optimizer)
        for leaf in leaves:
            mapping = getattr(leaf, "model_param_group_index_map", None)
            if mapping is not None and parameter in mapping:
                group_index = mapping[parameter][0]
                return leaf.optimizer.param_groups[group_index], leaf

        candidates = {id(parameter)}
        main_parameter = getattr(parameter, "main_param", None)
        if main_parameter is not None:
            candidates.add(id(main_parameter))
        for leaf in leaves:
            inner_optimizer = getattr(leaf, "optimizer", None)
            if inner_optimizer is None:
                continue
            for group in inner_optimizer.param_groups:
                if any(id(group_parameter) in candidates for group_parameter in group["params"]):
                    return group, leaf

        # DistributedOptimizer keeps globally aligned empty groups. On a DP
        # rank that owns no shard of this small gain, the original group index
        # still identifies the correct runtime group.
        if len(leaves) == 1 and fallback_index < len(leaves[0].optimizer.param_groups):
            return leaves[0].optimizer.param_groups[fallback_index], leaves[0]
        raise RuntimeError("Could not map the matched norm gain to a runtime optimizer group")

    def checked_setup_model_and_optimizer(*args: Any, **kwargs: Any) -> Any:
        # This call includes checkpoint restore. Inspecting after it returns is
        # important: optimizer state loading can restore param-group metadata.
        result = original_setup_model_and_optimizer(*args, **kwargs)
        _, optimizer, _ = result
        if optimizer is None:
            raise RuntimeError("Norm weight-decay check requires an optimizer")
        runtime_args = training_module.get_args()
        checkpoint_requested = runtime_args.load is not None
        checkpoint_iteration = int(getattr(runtime_args, "iteration", 0))
        if not checkpoint_requested:
            raise RuntimeError(
                "Norm weight-decay check requires --load so it can inspect a restored checkpoint"
            )

        local_matches: list[dict[str, Any]] = []
        for captured in captured_matches:
            group, leaf = find_runtime_group(
                optimizer,
                captured["parameter_object"],
                captured["group_index"],
            )
            local_matches.append(
                {
                    "rank": dist.get_rank(),
                    "parameter": captured["parameter"],
                    "shape": captured["shape"],
                    "group_index": captured["group_index"],
                    "configured_weight_decay": float(leaf.config.weight_decay),
                    "wd_mult": float(group.get("wd_mult", 1.0)),
                    "effective_weight_decay": float(group.get("weight_decay", 0.0)),
                    "lr_mult": float(group.get("lr_mult", 1.0)),
                    "lr": float(group.get("lr", 0.0)),
                    "checkpoint_load_path": runtime_args.load,
                    "checkpoint_iteration": checkpoint_iteration,
                }
            )

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

        return result

    optimizer_module._get_param_groups = checked_get_param_groups
    training_module.setup_model_and_optimizer = checked_setup_model_and_optimizer

    # Preserve the original Megatron CLI exactly; runpy executes its normal
    # __main__ path after the two optimizer hooks above are installed.
    sys.argv[0] = str(entrypoint)
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
