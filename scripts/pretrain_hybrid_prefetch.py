#!/usr/bin/env python3
"""Entrypoint wrapper: apply the data-staging monkeypatches (node-local mirror read-through,
windowed shuffle, sequential cyclic sampler, consumption cursor), then run Megatron's
pretrain_hybrid.py unchanged -- the HybridModel twin of scripts/pretrain_gpt_prefetch.py.

Every patch is a no-op unless its environment variable is set (OELLM_MIRROR_DIR,
OELLM_SHUFFLE_LANES, OELLM_CURSOR_FILE), so this is a safe drop-in for pretrain_hybrid.py.

With OELLM_PREFETCH_AUTOSTART=1, local rank 0 of every node also starts the node's prefetcher
(scripts/prefetch_warm.py) in the background, inside the training container. This replaces the
sbatch-level prefetcher launch of templates/marenostrum_mirror.sbatch, so it works with array
jobs, whose tasks share one script header but differ in data cache, data and parallelism.
The prefetcher's arguments come from this run's own Megatron arguments (data cache, data path
or per-split JSON, micro batch, TP / CP / PP for the data-parallel layout) and from the
PREFETCH_* / OELLM_* environment (config/data_staging/mirror_mn5.yaml: +data_staging=mirror_mn5). Its log goes
to <data cache>/prefetch_node<N>.log.
"""

import atexit
import os
import runpy
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MEGATRON = os.path.join(_REPO, "submodules", "Megatron-LM")
sys.path.insert(0, _MEGATRON)
sys.path.insert(0, _REPO)


def _arg(name: str, default=None, many: bool = False):
    """Value(s) of a Megatron CLI argument in sys.argv (``--name v`` / ``--name=v``)."""
    argv = sys.argv[1:]
    for i, tok in enumerate(argv):
        if tok == name:
            if not many:
                return argv[i + 1] if i + 1 < len(argv) else default
            vals = []
            for v in argv[i + 1:]:
                if v.startswith("--"):
                    break
                vals.append(v)
            return vals
        if tok.startswith(name + "="):
            v = tok.split("=", 1)[1]
            return v.split() if many else v
    return default


def _data_prefix() -> str | None:
    """The prefetcher's --prefix: the per-split JSON (``json:``) or the train blend's prefixes."""
    per_split = _arg("--per-split-data-args-path")
    if per_split:
        return "json:" + per_split
    paths = _arg("--train-data-path", many=True) or _arg("--data-path", many=True) or []
    if len(paths) > 1:
        try:  # weighted blend "w1 p1 w2 p2 ...": keep the prefixes
            float(paths[0])
            paths = paths[1::2]
        except ValueError:
            pass
    return ",".join(paths) or None


def _node_dp_ranks(node: int, local_world: int, world: int) -> tuple[list[int], int]:
    """Data-parallel ranks served by this node's GPUs under Megatron's default rank order
    (tp-cp-ep-dp-pp; for the dense / data grid: tp fastest, then cp, dp, pp)."""
    tp = int(_arg("--tensor-model-parallel-size", 1))
    cp = int(_arg("--context-parallel-size", 1))
    pp = int(_arg("--pipeline-model-parallel-size", 1))
    dp = world // (tp * cp * pp)
    ranks = range(node * local_world, (node + 1) * local_world)
    return sorted({(g // (tp * cp)) % dp for g in ranks}), dp


def _autostart_prefetcher() -> None:
    env = os.environ
    if env.get("OELLM_PREFETCH_AUTOSTART") != "1":
        return
    mirror = env.get("OELLM_MIRROR_DIR")
    if mirror:  # every local rank: the read-through reader needs the dir at dataset build time
        os.makedirs(mirror, exist_ok=True)
    if env.get("LOCAL_RANK", "0") != "0":
        return
    job = env.get("SLURM_JOB_ID")
    if env.get("OELLM_PREFETCH_CLEAN_SCRATCH") == "1" and job and os.path.isdir("/scratch/tmp"):
        # nodes are exclusive: node-local scratch of other (dead) jobs only fills the disk
        import shutil

        for d in os.listdir("/scratch/tmp"):
            if d != job:
                shutil.rmtree(os.path.join("/scratch/tmp", d), ignore_errors=True)
    cache, prefix = _arg("--data-cache-path"), _data_prefix()
    if not (mirror and cache and prefix):
        print(f"[prefetch] not started: mirror={mirror} cache={cache} prefix={prefix}", flush=True)
        return
    local_world = int(env.get("LOCAL_WORLD_SIZE", "4"))
    world = int(env.get("WORLD_SIZE", str(local_world)))
    node = int(env.get("RANK", "0")) // local_world
    dp_ranks, dp = _node_dp_ranks(node, local_world, world)
    cmd = [
        sys.executable, os.path.join(_REPO, "scripts", "prefetch_warm.py"),
        "--mode", env.get("PREFETCH_MODE", "lanes"),
        "--lanes", env.get("OELLM_SHUFFLE_LANES", "256"),
        "--lane-block-docs", env.get("PREFETCH_LANE_BLOCK", "32768"),
        "--threads", env.get("PREFETCH_THREADS", "8"),
        "--mirror-dir", mirror,
        "--prefix", prefix,
        "--cache-dir", cache,
        "--dp-size", str(dp),
        "--node-dp-ranks", ",".join(map(str, dp_ranks)),
        "--gpus-per-node", str(local_world),
        "--micro-batch-size", _arg("--micro-batch-size", "1"),
        "--wait-for-cache", env.get("PREFETCH_WAIT_FOR_CACHE", "1200"),
        "--budget-gb", env.get("PREFETCH_BUDGET_GB", "300"),
        "--passes", env.get("PREFETCH_PASSES", "1"),
    ]
    os.makedirs(cache, exist_ok=True)
    log = open(os.path.join(cache, f"prefetch_node{node}.log"), "a")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"[prefetch] node {node}: started pid {proc.pid} (dp ranks {dp_ranks} of {dp})",
          flush=True)

    def _stop():
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    atexit.register(_stop)


_autostart_prefetcher()

import oellm_autoexp.data_staging.megatron_patch  # noqa: F401,E402  (applies the patches on import)

if __name__ == "__main__":
    sys.argv[0] = os.path.join(_MEGATRON, "pretrain_hybrid.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
