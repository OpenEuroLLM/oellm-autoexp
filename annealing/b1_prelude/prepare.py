"""Render a standalone B1 annealing arm in an isolated run directory."""
import argparse
import hashlib
import json
import pickletools
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from omegaconf import OmegaConf


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
MODEL_CONFIG = "experiments/oellm_32b_dense/b1_prelude_annealing"
TOKENS_PER_UPDATE = 4096 * 4096


def checkpoint_files(seed):
    metadata = seed / ".metadata"
    shards = {
        value for _, value, _ in pickletools.genops(metadata.read_bytes())
        if isinstance(value, str) and value.endswith(".distcp")
    }
    if not shards:
        raise ValueError(f"No indexed checkpoint shards in {metadata}")
    for name in shards:
        path = seed / name
        if Path(name).name != name or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or invalid checkpoint shard: {path}")
    return len(shards)


def checkpoint_gate(root):
    recipe = json.loads((root / "recipe.json").read_text())
    if (root / "STOP").exists():
        raise ValueError(f"Run held by {root / 'STOP'}")
    iterations = []
    for name in ("checkpoints", "checkpoints_rolling"):
        tree = root / "training_ckpts" / name
        tracker = tree / "latest_checkpointed_iteration.txt"
        if not tracker.is_file() and name == "checkpoints_rolling":
            continue
        iteration = int(tracker.read_text().strip())
        directory = tree / f"iter_{iteration:07d}"
        if directory.is_symlink() and directory.resolve() != Path(recipe["seed"]):
            raise ValueError(f"Unexpected checkpoint symlink: {directory}")
        checkpoint_files(directory)
        iterations.append(iteration)
    if not recipe["seed_iteration"] <= max(iterations) < recipe["endpoint"]:
        raise ValueError("Checkpoint is outside the executable annealing interval")


def capped_events(job, root, limit):
    for field in ("log_events", "state_events"):
        for event in job.get(field, []):
            if event.get("action", {}).get("class_name") != "RestartAction":
                continue
            conditions = [
                {"class_name": "MaxAttemptsCondition", "max_attempts": limit},
                {"class_name": "NotCondition", "condition": {
                    "class_name": "FileExistsCondition", "path": str(root / "STOP"),
                }},
            ]
            if event.get("condition"):
                conditions.append(event["condition"])
            conditions.append({"class_name": "CommandCondition", "command": [
                sys.executable, str(Path(__file__).resolve()), "--check-resume", str(root),
            ]})
            event["condition"] = {"class_name": "AndCondition", "conditions": conditions}


def prepare(args):
    root = args.root.resolve()
    seed = args.seed.resolve()
    if root.exists() or not seed.name.startswith("iter_"):
        raise ValueError("Use a new run directory and an explicit iter_N checkpoint")
    if args.tokens <= 0 or args.nodes < 4 or 4096 % (args.nodes // 4) or args.nodes % 4:
        raise ValueError("Require a positive budget and TP4/PP4-compatible DP dividing GBS")
    if not 1 <= args.attempts <= 32:
        raise ValueError("Specify a finite allocation count from 1 to 32")
    start = int(seed.name.removeprefix("iter_"))
    updates = (args.tokens + TOKENS_PER_UPDATE - 1) // TOKENS_PER_UPDATE
    endpoint = start + updates
    if args.smoke and updates < 3:
        raise ValueError("Smoke needs an annealing window of at least three updates")
    shards = checkpoint_files(seed)
    source_mix = PACKAGE / "datamix" / (args.mix + ".txt")
    fields = shlex.split(source_mix.read_text())
    if len(fields) != 588 or len(set(fields[1::2])) != 294:
        raise ValueError("Unexpected blend coverage")
    for prefix in fields[1::2]:
        for extension in (".bin", ".idx"):
            if not Path(prefix + extension).is_file():
                raise ValueError(f"Dataset file missing: {prefix}{extension}")

    root.mkdir(parents=True)
    shutil.copytree(REPO / "config", root / "config", symlinks=True)
    shutil.copyfile(source_mix, root / "datamix.txt")
    permanent = root / "training_ckpts/checkpoints"
    permanent.mkdir(parents=True)
    (permanent / seed.name).symlink_to(seed, target_is_directory=True)
    (permanent / "latest_checkpointed_iteration.txt").write_text(f"{start}\n")
    logs = root / "training_core/logs"
    logs.mkdir(parents=True)
    cache = args.cache.resolve() if args.cache else root / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    name = root.name
    job = dict(name=name, est_steps=3 if args.smoke else updates)
    if args.smoke:
        job.update(log_events=[], state_events=[])
    else:
        policy = root / "config/job/auto_restart_ckptreset_faultscan.yaml"
        events = OmegaConf.to_container(OmegaConf.load(policy), resolve=False)
        capped_events(events, root, args.attempts)
        job.update({key: events[key] for key in ("log_events", "state_events") if key in events})
    megatron = {
        "train_samples": endpoint * 4096,
        "lr_decay_samples": endpoint * 4096,
        "lr_wsd_decay_samples": updates * 4096,
        "phase_transition_iterations": str(start),
        "exit_interval": endpoint,
        "data_args_path": str(root / "datamix.txt"),
        "data_cache_path": str(cache),
        "start_eval_at_iter": start,
        "tensorboard_dir": str(root / "tensorboard"),
        "wandb_save_dir": str(root / "wandb"),
    }
    if args.smoke:
        smoke_end = start + 3
        megatron.update(
            phase_transition_iterations=f"{start},{smoke_end}" if smoke_end < endpoint else str(start),
            exit_interval=smoke_end,
            save=None, non_persistent_save_interval=None, exit_signal_handler=False,
            eval_iters=1, eval_global_batch_size=16, eval_micro_batch_size=1,
            log_interval=1, diagnostics_interval=100,
            wgrad_deferral_limit=2, ddp_bucket_size=128000000,
        )
    override = {
        "defaults": ["/" + MODEL_CONFIG, "_self_"],
        "aux": {"run_root": str(root)},
        "job": job,
        "backend": {
            "aux": {"active_nodes": args.nodes}, "megatron": megatron,
            "launcher_script": str(PACKAGE / "training.py"),
        },
        "slurm": {
            "template_path": str(root / "config/annealing_lock.sbatch"),
            "sbatch": {"time": "00:10:00" if args.smoke else args.walltime,
                       "account": args.account, "signal": None if args.smoke else "B:USR1@240"},
            "env": {"HF_HOME": str(root / "hf-cache")},
        },
    }
    # Preserve the complete production relay/mount/GPU contract, adding only
    # a shared cache-writer lease around the allocation's existing launcher.
    template = (REPO / 'templates/jupiter_relay.sbatch').read_text()
    marker = 'module purge\n'
    if template.count(marker) != 1:
        raise ValueError('Unknown production relay template')
    lock = 'exec 8>' + shlex.quote(str(cache / '.writer.lock')) + '\nflock -n 8 || exit 75\n'
    (root / 'config/annealing_lock.sbatch').write_text(template.replace(marker, lock + marker))
    config = root / "config/annealing_arm.yaml"
    OmegaConf.save(override, config)
    argv = [
        sys.executable, str(REPO / "scripts/run_autoexp.py"), "--config-dir", str(root / "config"),
        "--config-name", "annealing_arm", "--monitor-state-dir", str(root / "monitor"),
    ]
    if args.smoke:
        argv.append("--no-monitor")
    script = root / "launch.sh"
    preflight = shlex.join([sys.executable, str(Path(__file__).resolve()), "--check-resume", str(root)])
    cache_check = shlex.join([sys.executable, str(PACKAGE / 'cache.py'), 'check', '--root', str(root)])
    script.write_text("#!/bin/bash\nset -euo pipefail\ncd " + shlex.quote(str(REPO)) +
                      "\nexport PYTHONPATH=" + shlex.quote(str(REPO)) +
                      "\n" + preflight +
                      "\nexec 9>" + shlex.quote(str(root / "launch.lock")) +
                      "\nflock -n 9" +
                      "\nif [[ ${1:-} != --dry-run && -d " + shlex.quote(str(root / "monitor")) +
                      " ]]; then echo 'Use the existing monitor session for this run' >&2; exit 1; fi" +
                      "\nif [[ ${1:-} != --dry-run ]]; then " + cache_check + "; fi" +
                      "\nexec " + shlex.join(argv) + ' "$@"\n')
    script.chmod(0o755)
    receipt = {
        "seed": str(seed), "seed_shards": shards, "seed_iteration": start,
        "seed_metadata_sha256": hashlib.sha256((seed / ".metadata").read_bytes()).hexdigest(),
        "mix": args.mix, "datamix_sha256": hashlib.sha256(source_mix.read_bytes()).hexdigest(),
        "requested_tokens": args.tokens, "updates": updates, "actual_tokens": updates * TOKENS_PER_UPDATE,
        "endpoint": endpoint, "smoke": args.smoke, "nodes": args.nodes,
        "maximum_allocations": 1 if args.smoke else args.attempts,
        "allocation_walltime": "00:10:00" if args.smoke else args.walltime,
        "fork_validation_cursor": "reset_on_seed_restore",
        "source": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
    }
    (root / "recipe.json").write_text(json.dumps(receipt, indent=2) + "\n")
    subprocess.run([str(script), "--dry-run"], check=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--check-resume":
        checkpoint_gate(Path(sys.argv[2]))
        raise SystemExit(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--mix", choices=("dominant1", "dominant2"), required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--walltime", default="12:00:00")
    parser.add_argument("--account", default="e-ext-2025e02-108")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cache", type=Path, help="Reuse a compatible cache; exact coverage is verified before launch")
    prepare(parser.parse_args())
