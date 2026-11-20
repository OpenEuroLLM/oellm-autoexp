"""Run native Megatron training with a fresh validation cursor at the data fork."""
import importlib
import json
from pathlib import Path
import runpy
import sys


def fork_validation_cursor(args, recipe):
    """Return the previous cursor on a seed restore, or None on an arm resume."""
    fork = recipe["seed_iteration"]
    if not fork <= args.iteration < recipe["endpoint"]:
        raise ValueError(f"Checkpoint iteration {args.iteration} is outside the annealing arm")
    if args.finetune or args.override_ckpt_iteration is not None:
        raise ValueError("Annealing requires the actual full-state checkpoint iteration")
    if args.iteration != fork:
        return None
    link = Path(args.load) / f"iter_{fork:07d}"
    if not link.is_symlink() or link.resolve(strict=True) != Path(recipe["seed"]):
        raise ValueError("Annealing fork checkpoint does not match the arm's seed link")
    if not args.train_samples or args.start_eval_at_iter != fork:
        raise ValueError("Annealing requires a sample-based schedule and evaluation starting at the fork")
    previous = args.consumed_valid_samples
    args.consumed_valid_samples = 0
    return previous


def main():
    megatron = Path(__file__).resolve().parents[2] / "submodules/Megatron-LM"
    sys.path.insert(0, str(megatron))
    training = importlib.import_module("megatron.training.training")
    native_setup = training.setup_model_and_optimizer

    def setup(*args, **kwargs):
        result = native_setup(*args, **kwargs)
        config = training.get_args()
        load = Path(config.load)
        if load.parts[-2:] != ("training_ckpts", "checkpoints"):
            raise ValueError("Annealing requires its own checkpoint tree")
        recipe = json.loads((load.parents[1] / "recipe.json").read_text())
        previous = fork_validation_cursor(config, recipe)
        if previous is not None:
            training.print_rank_0(
                f"Annealing fork validation cursor at iteration {config.iteration}: {previous} -> 0"
            )
        return result

    training.setup_model_and_optimizer = setup
    try:
        runpy.run_path(str(megatron / "pretrain_gpt.py"), run_name="__main__")
    finally:
        training.setup_model_and_optimizer = native_setup


if __name__ == "__main__":
    main()
