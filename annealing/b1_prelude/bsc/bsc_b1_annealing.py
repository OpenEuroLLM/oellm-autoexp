#!/usr/bin/env python3
"""Audit BSC data and prepare the two bounded B1 annealing validations."""
import argparse
import fcntl
import hashlib
import json
import importlib.util
import re
import shutil
import sys
import time
import subprocess
import shlex
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe(prefix):
    with Path(str(prefix) + ".idx").open("rb") as index, Path(str(prefix) + ".bin").open("rb") as binary:
        binary.read(1)
        return [
            Path(index.name).stat().st_size,
            Path(binary.name).stat().st_size,
            hashlib.sha256(index.read(256)).hexdigest(),
        ]


def audit(expected, roots):
    def check(item):
        suffix, baseline = item
        if Path(suffix).is_absolute() or ".." in Path(suffix).parts:
            raise ValueError(f"Unsafe prefix suffix: {suffix}")
        matches, different = [], {}
        for root in roots:
            path = str(Path(root) / suffix)
            try:
                facts = probe(path)
            except (FileNotFoundError, PermissionError):
                continue
            if facts == baseline:
                matches.append(path)
            else:
                different[path] = facts
        return suffix, {"expected": baseline, "matching": matches, "different": different}

    with ThreadPoolExecutor(max_workers=8) as pool:
        prefixes = dict(pool.map(check, expected.items()))
    matches = {k: v["matching"][0] for k, v in prefixes.items() if v["matching"]}
    return {
        "scope": "Both files open, sizes and first 256 index bytes match. Payload identity is separate.",
        "matches": matches,
        "required": len(prefixes),
        "missing_bytes": sum(sum(v["expected"][:2]) for k, v in prefixes.items() if k not in matches),
        "prefixes": prefixes,
    }


def remap(text, reference_root, matches):
    fields = shlex.split(text)
    if not fields or len(fields) % 2:
        raise ValueError("Invalid weight/prefix pairs")
    roots = [Path(reference_root)] if isinstance(reference_root, str) else list(map(Path, reference_root))
    rows, seen = [], set()
    for weight, prefix in zip(fields[::2], fields[1::2]):
        if not Decimal(weight).is_finite() or Decimal(weight) <= 0:
            raise ValueError(f"Invalid weight: {weight}")
        sources = [root for root in roots if Path(prefix).is_relative_to(root)]
        if len(sources) != 1:
            raise ValueError(f"Expected one reviewed JUPITER FLAG root: {prefix}")
        suffix = str(Path(prefix).relative_to(sources[0]))
        new = matches.get(suffix)
        if new is None:
            raise ValueError(f"BSC prefix missing: {suffix}")
        if not Path(new).is_absolute() or len(new.split()) != 1:
            raise ValueError(f"Invalid destination prefix: {new}")
        if new in seen:
            raise ValueError(f"BSC prefix duplicate: {new}")
        seen.add(new)
        rows.append(f"{weight} {new}\n")
    return "".join(rows)


def smoke_qos(nodes):
    # 32 nodes retain the validated shape. 64 provides H100 timing margin.
    if nodes not in (32, 64):
        raise ValueError("Choose 32 or 64 nodes to retain the 16-sample validation batch")
    return "acc_ehpc"


def write(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def bind_arguments(argv, replacements, remove, remove_switches=()):
    bound = []
    used = set()
    i = 0
    while i < len(argv):
        flag = argv[i]
        if flag in remove_switches:
            i += 1
        elif flag in replacements or flag in remove:
            if i + 1 >= len(argv) or argv[i + 1].startswith("--") or flag in used:
                raise ValueError(f"Expected one value for {flag}")
            if flag in replacements:
                bound.extend((flag, str(replacements[flag])))
            used.add(flag)
            i += 2
        else:
            bound.append(flag)
            i += 1
    if replacements.keys() - used:
        raise ValueError("Missing a required binding argument")
    if any(v.startswith(("/e/", "/p/")) or "$SLURM" in v for v in bound):
        raise ValueError("Found unbound JUPITER argument")
    return bound


def batch(run, source, image, mode, nodes):
    if mode == "smoke":
        qos, time, workers = smoke_qos(nodes), "00:10:00", 4
    elif mode in ("cache", "probe") and nodes == 1:
        qos, time, workers = "acc_debug", ("00:30:00" if mode == "cache" else "00:05:00"), (1 if mode == "cache" else 4)
    else:
        raise ValueError("Invalid ACC stage")
    if not all(p.is_absolute() and re.fullmatch(r"[A-Za-z0-9_./+-]+", str(p)) for p in (run, source, image)):
        raise ValueError("Use absolute shell-safe run, source and image paths")
    lock = "" if mode == "probe" else f"exec 9>{run}/cache/.writer.lock\nflock -n 9\n"
    return f"""#!/bin/bash
#SBATCH --account=ehpc533
#SBATCH --partition=acc
#SBATCH --qos={qos}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=160
#SBATCH --threads-per-core=2
#SBATCH --exclusive
#SBATCH --time={time}
#SBATCH --job-name=b1-{run.name}-{mode}
#SBATCH --output={run}/{mode}-%j.log
set -euo pipefail
test ! -e {run}/STOP
test ! -e {run.parent}/STOP
{lock}export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 30000))
exec srun --kill-on-bad-exit=1 --mpi=none --gpu-bind=none --cpu-bind=none bash {run}/bsc_b1_annealing_node.sh {mode} {run} {source} {image} {workers}
"""


def load_upstream(source, name):
    entry = source / "annealing" / "b1_prelude" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def submit_probe(runtime, source, image):
    if (runtime / "STOP").exists() or (runtime / "probe.intent").exists() or (runtime / "probe.job").exists():
        raise ValueError("The runtime is stopped or already has a probe intent")
    for name in ("bsc_b1_annealing_node.sh", "bsc_b1_annealing_worker.py"):
        source_file = Path(__file__).with_name(name)
        destination = runtime / name
        if destination.exists() and digest(destination) != digest(source_file):
            raise ValueError(f"Runtime file changed: {destination}")
        if not destination.exists():
            shutil.copyfile(source_file, destination)
    script = runtime / "probe.sbatch"
    text = batch(runtime, source, image, "probe", 1)
    if script.exists() and script.read_text() != text:
        raise ValueError("Use a fresh runtime for a different probe")
    script.write_text(text)
    subprocess.run(["bash", "-n", str(script)], check=True)
    subprocess.run(["sbatch", "--test-only", str(script)], check=True)
    with (runtime / "probe.intent").open("x") as stream:
        json.dump({"script": str(script), "sha256": digest(script)}, stream)
    job = subprocess.check_output(["sbatch", "--parsable", str(script)], text=True).strip().split(";")[0]
    if not job.isdecimal():
        raise ValueError("Uncertain probe submission, reconcile the retained intent")
    (runtime / "probe.job").write_text(job + "\n")
    print(job)


def prepare(root, bundle, inventory, source, image, seed, nodes):
    smoke_qos(nodes)
    config = read(bundle / "recipe.json")
    mappings = read(inventory)["matches"]
    expected = read(bundle / "expected.json")
    if set(mappings) != set(expected) or len(mappings) != 294:
        raise ValueError("Require all 294 source prefixes before preparing")
    for name, sha in config["tokenizer_files"].items():
        if digest(Path(config["tokenizer"]) / name) != sha:
            raise ValueError(f"Tokenizer file changed: {name}")
    # The inventory records paths from different roots. Verify the chosen exact
    # path for each prefix so a changed path cannot silently be substituted.
    with ThreadPoolExecutor(max_workers=8) as pool:
        actual = dict(zip(mappings, pool.map(probe, mappings.values())))
    if actual != expected:
        raise ValueError("Chosen input pairs differ from the JUPITER geometry receipt")
    if seed.name != "iter_0416000":
        raise ValueError("Require the validated full-state checkpoint iteration")
    vectors = read(bundle / "vectors.json")
    for name, sha in config["input_hashes"].items():
        if digest(bundle / name) != sha:
            raise ValueError(f"Input bundle file changed: {name}")
    image_stat = image.stat()
    mixes = {arm: remap((bundle / f"{arm}.txt").read_text(), config["jupiter_flag_roots"], mappings)
             for arm in ("dominant1", "dominant2")}
    root.mkdir(mode=0o700, parents=False, exist_ok=False)
    for arm in ("dominant1", "dominant2"):
        run = root / arm
        run.mkdir()
        datamix = run / "datamix.txt"
        datamix.write_text(mixes[arm])
        cache = run / "cache"
        cache.mkdir()
        permanent = run / "training_ckpts/checkpoints"
        permanent.mkdir(parents=True)
        (permanent / seed.name).symlink_to(seed)
        (permanent / "latest_checkpointed_iteration.txt").write_text("416000\n")
        remove = {"--wandb-entity", "--wandb-exp-name", "--wandb-project", "--wandb-save-dir"}
        common = {"--data-args-path": datamix, "--data-cache-path": cache, "--tokenizer-model": config["tokenizer"]}
        for mode in ("cache", "smoke"):
            replacements = dict(common)
            if mode == "smoke":
                replacements.update({"--load": permanent, "--tensorboard-dir": run / "tensorboard",
                    "--non-persistent-global-ckpt-dir": run / "training_ckpts/checkpoints_rolling",
                    "--distributed-timeout-minutes": 8})
            else:
                replacements["--distributed-timeout-minutes"] = 20
            # BSC H100 nodes in this image cannot initialize TE CUDA multicast.
            switches = {"--tp-comm-overlap"} if mode == "smoke" else ()
            write(run / f"{mode}-argv.json", bind_arguments(vectors[mode]["argv"], replacements, remove, switches))
            path = run / f"{mode}.sbatch"
            path.write_text(batch(run, source, image, mode, nodes if mode == "smoke" else 1))
            subprocess.run(["bash", "-n", str(path)], check=True)
        settings = config["cache_target"]
        write(run / "cache-settings.json", {"target": dict(settings, data_args_path=str(datamix),
              data_cache_path=str(cache), tokenizer_model=config["tokenizer"]), "samples": config["cache_samples"]})
        for filename in ("bsc_b1_annealing_node.sh", "bsc_b1_annealing_worker.py"):
            shutil.copyfile(Path(__file__).with_name(filename), run / filename)
        files = ["cache.sbatch", "smoke.sbatch", "cache-argv.json", "smoke-argv.json", "cache-settings.json",
                 "datamix.txt", "bsc_b1_annealing_node.sh", "bsc_b1_annealing_worker.py"]
        write(run / "recipe.json", {"seed": str(seed), "seed_iteration": 416000, "endpoint": 417193,
            "seed_metadata_sha256": config["seed_metadata_sha256"], "source": str(source),
            "source_revision": config["autoexp"], "megatron_revision": config["megatron"],
            "image": str(image), "image_bytes": image_stat.st_size, "image_mtime_ns": image_stat.st_mtime_ns,
            "tokenizer": config["tokenizer"], "tokenizer_files": config["tokenizer_files"],
            "smoke_nodes": nodes, "smoke_seconds": 600, "cache_seconds": 1800, "mix": arm,
            "pins": {x: digest(run / x) for x in files}})


def accounting(job):
    if not re.fullmatch(r"[0-9]+", job):
        raise ValueError("Invalid accounting job ID")
    log = subprocess.check_output(["sacct", "-X", "-nP", "-j", job,
        "-o", "JobIDRaw,State,ExitCode,ElapsedRaw,AllocNodes,AllocTRES,TimelimitRaw"], text=True)
    for line in log.splitlines():
        fields = line.rstrip("|").split("|")
        if len(fields) == 7 and fields[0] == job:
            _, state, code, elapsed, nodes, tres, limit = fields
            return dict(job=job, state=state, exit_code=code, seconds=int(elapsed),
                        nodes=int(nodes), tres=tres, limit_minutes=int(limit))
    return {"job": job, "state": "UNKNOWN"}


def check_pins(run, recipe):
    if any((parent / "STOP").exists() for parent in (run, run.parent)):
        raise ValueError("Run held by STOP")
    for filename, sha in recipe["pins"].items():
        if digest(run / filename) != sha:
            raise ValueError(f"Pinned input changed: {filename}")
    for filename, sha in recipe["tokenizer_files"].items():
        if digest(Path(recipe["tokenizer"]) / filename) != sha:
            raise ValueError(f"Tokenizer file changed: {filename}")
    stat = Path(recipe["image"]).stat()
    if (stat.st_size, stat.st_mtime_ns) != (recipe["image_bytes"], recipe["image_mtime_ns"]):
        raise ValueError("BSC image identity changed")


def result(run, mode):
    recipe = read(run / "recipe.json")
    receipt = read(run / f"{mode}-submitted.json")
    job = receipt["job"]
    row = accounting(job)
    if (row["state"], row.get("exit_code")) != ("COMPLETED", "0:0"):
        return row
    log_dir = Path(receipt["reused_from"]) if mode == "cache" and "reused_from" in receipt else run
    log = (log_dir / f"{mode}-{job}.log").read_text()
    source = Path(recipe["source"])
    if mode == "cache":
        if not re.search(r"exiting program at iteration\s+1\b", log):
            raise ValueError("Cache job lacks one-update completion evidence")
        output = subprocess.check_output(
            [sys.executable, str(source / "scripts/check_data_cache.py"), str(run / "cache")], text=True)
        proof = load_upstream(source, "cache").coverage(run, read(run / "cache-settings.json"))
        row.update(cache_coverage=proof, checker=output)
    else:
        if row["limit_minutes"] != 10 or row["nodes"] != recipe["smoke_nodes"]:
            raise ValueError("Unexpected smoke allocation shape")
        row.update(load_upstream(source, "validate").validate_log(log, require_cursor_reset=True))
    return row


def reuse_cache(run, prior):
    """Bind a fresh arm to an identical cache from a completed preparation."""
    stage_files = (f"{mode}-{kind}.json" for mode in ("cache", "smoke") for kind in ("intent", "submitted"))
    if any((run / name).exists() for name in stage_files):
        raise ValueError("The new arm already has a submission intent or receipt")
    current, original = (read(arm / "recipe.json") for arm in (run, prior))
    check_pins(run, current)
    check_pins(prior, original)
    for key in ("source_revision", "megatron_revision", "tokenizer", "tokenizer_files"):
        if current.get(key) != original.get(key):
            raise ValueError(f"Cache provenance differs: {key}")
    settings = [read(arm / "cache-settings.json") for arm in (run, prior)]
    for item in settings:
        for key in ("data_args_path", "data_cache_path"):
            item["target"].pop(key, None)
    if settings[0] != settings[1] or digest(run / "datamix.txt") != digest(prior / "datamix.txt"):
        raise ValueError("Cache settings or mix differ")
    for mode in ("cache", "smoke"):
        old_receipt = prior / f"{mode}-submitted.json"
        if old_receipt.exists() and accounting(read(old_receipt)["job"])["state"] in ACTIVE:
            raise ValueError("The prior arm still has an active writer or consumer")
    receipt = read(prior / "cache-submitted.json")
    if "reused_from" in receipt or (prior / "cache").is_symlink():
        raise ValueError("Use the original cache allocation")
    if "cache_coverage" not in result(prior, "cache"):
        raise ValueError("Prior cache lacks successful coverage")
    destination = run / "cache"
    if destination.is_symlink():
        if destination.resolve() != prior / "cache":
            raise ValueError("New cache already points elsewhere")
    else:
        if destination.exists():
            destination.rmdir()
        destination.symlink_to(prior / "cache", target_is_directory=True)
    write(run / "cache-submitted.json", dict(receipt, reused_from=str(prior)))


def admission(run, nodes):
    if nodes not in (128, 256):
        raise ValueError("Choose 128 or 256 nodes for large-node admission checks")
    check_pins(run, read(run / "recipe.json"))
    subprocess.run(["sbatch", "--test-only", f"--nodes={nodes}", "--qos=acc_xehpc",
                    str(run / "smoke.sbatch")], check=True)


def submit(run, mode, dry_run=False):
    recipe = read(run / "recipe.json")
    check_pins(run, recipe)
    if (run / f"{mode}-submitted.json").exists() or (run / f"{mode}-intent.json").exists():
        raise ValueError("This stage already has a submission receipt or intent")
    if mode == "smoke":
        cache = result(run, "cache")
        if "cache_coverage" not in cache:
            raise ValueError("Smoke awaits completed and verified cache preparation")
        seed = Path(recipe["seed"])
        marker = read(seed / ".hpc-transfer-verified.json")
        if marker.get("result", {}).get("sha256_verified") is not True or digest(seed / ".metadata") != recipe["seed_metadata_sha256"]:
            raise ValueError("Smoke awaits the SHA-verified B1 checkpoint")
        write(run / "cache-proof.json", cache)
    script = run / f"{mode}.sbatch"
    subprocess.run(["sbatch", "--test-only", str(script)], check=True)
    if dry_run:
        return
    with (run / f"{mode}-intent.json").open("x") as stream:
        json.dump({"script": str(script), "sha256": digest(script)}, stream)
    raw = subprocess.check_output(["sbatch", "--parsable", str(script)], text=True)
    job = raw.strip().split(";")[0]
    if not job.isdigit():
        raise ValueError("Uncertain submission, reconcile the retained intent")
    write(run / f"{mode}-submitted.json", {"job": job, "script_sha256": digest(script)})
    print(job)


ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "UNKNOWN", "SUSPENDED"}


def progress(run):
    status = {}
    for mode in ("cache", "smoke"):
        submitted = run / f"{mode}-submitted.json"
        if submitted.is_file():
            status[mode] = accounting(read(submitted)["job"])
        elif (run / f"{mode}-intent.json").exists():
            raise ValueError(f"Uncertain {mode} submission for {run.name}; inspect the intent")
    return status


def advance(runtime):
    """Submit each arm once, in dependency order. A failed stage holds its arm."""
    if (runtime / "STOP").exists():
        return {"status": "stopped"}
    probe_job = (runtime / "probe.job").read_text().strip()
    probe = accounting(probe_job)
    result_data = {"probe": probe, "status": "waiting"}
    if probe["state"] in ACTIVE:
        return result_data
    if (probe["state"], probe.get("exit_code")) != ("COMPLETED", "0:0"):
        raise ValueError(f"FP8 runtime probe failed: {probe}")
    if '"BSC_B1_RUNTIME": "passed"' not in (runtime / f"probe-{probe_job}.log").read_text():
        raise ValueError("Missing successful runtime evidence")
    runs = [runtime / "arms" / mix for mix in ("dominant1", "dominant2")]
    result_data["arms"] = {run.name: progress(run) for run in runs}
    cache_running = any(status.get("cache", {}).get("state") in ACTIVE for status in result_data["arms"].values())
    for run in runs:
        status = result_data["arms"][run.name]
        if (run / "STOP").exists():
            continue
        if "cache" not in status and not cache_running:
            submit(run, "cache")
            cache_running = True
            status.update(progress(run))
        if status.get("cache", {}).get("state") == "COMPLETED" and "smoke" not in status:
            cache_proof = run / "cache-proof.json"
            if not cache_proof.exists():
                proof = result(run, "cache")
                if "cache_coverage" not in proof:
                    raise ValueError(f"Cache proof failed for {run.name}")
                write(cache_proof, proof)
            status["cache"]["coverage_verified"] = True
            seed = Path(read(run / "recipe.json")["seed"])
            if (seed / ".hpc-transfer-verified.json").is_file():
                submit(run, "smoke")
                status.update(progress(run))
        if status.get("smoke", {}).get("state") == "COMPLETED":
            proof = result(run, "smoke")
            if "steps" in proof:
                write(run / "smoke-proof.json", proof)
                status["smoke"] = proof
    if all("steps" in status.get("smoke", {}) for status in result_data["arms"].values()):
        result_data["status"] = "passed"
    elif all((run / "STOP").exists() or any(row["state"] not in ACTIVE | {"COMPLETED"}
             for row in result_data["arms"][run.name].values()) or "steps" in result_data["arms"][run.name].get("smoke", {})
             for run in runs):
        result_data["status"] = "held"
    return result_data


def watch(runtime, hours):
    if not 0 < hours <= 168:
        raise ValueError("Watch duration must be between zero and 168 hours")
    with (runtime / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + hours * 3600
        while time.monotonic() < deadline:
            try:
                state = advance(runtime)
            except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
                write(runtime / "controller-error.json", {"status": "held", "error": str(error)})
                raise
            write(runtime / "status.json", state)
            if state["status"] != "waiting":
                print(json.dumps(state, indent=2))
                return
            time.sleep(45)
        print("Observation window ended. Resume watch to observe remaining allocations.")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    audit_cmd = sub.add_parser("audit")
    audit_cmd.add_argument("expected", type=Path)
    audit_cmd.add_argument("roots", nargs="+", type=Path)
    mix_cmd = sub.add_parser("remap")
    mix_cmd.add_argument("mixture", type=Path)
    mix_cmd.add_argument("inventory", type=Path)
    mix_cmd.add_argument("--reference-root", required=True, action="append")
    probe_cmd = sub.add_parser("probe")
    for key in ("root", "source", "image"):
        probe_cmd.add_argument("--" + key, type=Path, required=True)
    prep = sub.add_parser("prepare")
    for key in ("root", "bundle", "inventory", "source", "image", "seed"):
        prep.add_argument("--" + key, type=Path, required=True)
    prep.add_argument("--nodes", type=int, default=64)
    reuse = sub.add_parser("reuse-cache")
    reuse.add_argument("run", type=Path)
    reuse.add_argument("prior", type=Path)
    admit = sub.add_parser("admission")
    admit.add_argument("run", type=Path)
    admit.add_argument("--nodes", choices=(128, 256), required=True, type=int)
    for action in ("submit", "result"):
        command = sub.add_parser(action)
        command.add_argument("run", type=Path)
        command.add_argument("mode", choices=("cache", "smoke"))
        if action == "submit":
            command.add_argument("--dry-run", action="store_true")
    watcher = sub.add_parser("watch")
    watcher.add_argument("runtime", type=Path)
    watcher.add_argument("--hours", type=float, default=168)
    args = parser.parse_args()
    if args.action == "audit":
        print(json.dumps(audit(read(args.expected), args.roots), indent=2))
    elif args.action == "remap":
        print(remap(args.mixture.read_text(), args.reference_root, read(args.inventory)["matches"]), end="")
    elif args.action == "probe":
        submit_probe(args.root.resolve(), args.source.resolve(), args.image.resolve())
    elif args.action == "prepare":
        prepare(args.root.resolve(), args.bundle, args.inventory, args.source, args.image, args.seed, args.nodes)
    elif args.action == "reuse-cache":
        reuse_cache(args.run.resolve(), args.prior.resolve())
    elif args.action == "admission":
        admission(args.run.resolve(), args.nodes)
    elif args.action == "watch":
        watch(args.runtime.resolve(), args.hours)
    elif args.action == "submit":
        submit(args.run.resolve(), args.mode, args.dry_run)
    else:
        print(json.dumps(result(args.run.resolve(), args.mode), indent=2))


if __name__ == "__main__":
    main()
