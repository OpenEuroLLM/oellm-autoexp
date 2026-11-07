"""FLAG lm-eval rows (config/eval_tasks/flag_evals.yaml, half "vllm", suite lm-eval-harness) on
MareNostrum 5: ONE vLLM engine per node, the selected rows one after another (the JUPITER
launcher starts a fresh engine per row). Same harness calls as oellm-eval's template.sbatch
(v0.01: --tasks <task> --num_fewshot <n> --batch_size auto --log_samples
--confirm_run_unsafe_code --include_path <custom_lm_eval_tasks>, model args
pretrained=<export>,trust_remote_code=True,dtype=bfloat16); results in lm-eval's own layout.

    python scripts/korbi/flag_eval/flag_lmeval.py --export <hf export> --out <dir> \
        --names vllm/arc_challenge/25 vllm/piqa/10 ... [--shard 0 --num-shards 4] [--limit 20]

A row with <out>/<export name>[_limit<l>]/done/<task>_<n_shot>.ok is skipped; a failing row is logged and
the next one runs (exit status 1 at the end). <out>/<export name>/summary_shard<i>.tsv: one line
per (row, metric).

A name with ``@<k>of<K>`` (e.g. vllm/mmlu/5@1of3) is part k of K of the row: every K-th document
of its leaf tasks (lmeval_parts.py). Its per-document metrics go to parts/<task>_<n_shot>/; the part
that completes the set merges them with lm-eval's aggregation into the row's results and marker."""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import yaml

from scripts.korbi.flag_eval import lmeval_parts

REPO = Path(__file__).resolve().parents[3]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--names", nargs="+", help="flag_evals.yaml names half/task/n_shot")
    p.add_argument("--queue", help="flag_queue.py queue dir: evaluate its items until it is empty "
                                   "(one engine for all of them; instead of --names)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--dp", type=int, default=1,
                   help="node-local DP frontends; TP1/DP4/EP runs pure expert parallelism")
    p.add_argument("--ep", type=int, default=1, help="enable_expert_parallel (needed for the MoE bound)")
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--max-num-batched-tokens", type=int, default=None)
    p.add_argument("--request-batch-size", default="auto")
    p.add_argument("--response-cache", action="store_true",
                   help="durable per-request SQLite cache; requires a finite request batch size")
    p.add_argument("--limit", type=float, default=None, help="test runs only (lm-eval --limit)")
    p.add_argument("--suite", default=str(REPO / "config/eval_tasks/flag_evals.yaml"))
    p.add_argument("--include-path",
                   default=str(REPO / "submodules/oellm-eval/oellm/resources/custom_lm_eval_tasks"))
    a = p.parse_args(argv)
    if bool(a.names) == bool(a.queue):
        p.error("give --names or --queue")
    if a.dp < 1 or a.tp < 1:
        p.error("TP and DP sizes must be positive")
    if a.dp > 1 and not a.ep:
        p.error("node-local DP requires expert parallelism")
    if a.request_batch_size != "auto":
        try:
            if int(a.request_batch_size) < 1:
                raise ValueError()
        except ValueError:
            p.error("request batch size must be auto or a positive integer")
    if a.response_cache and a.request_batch_size == "auto":
        p.error("--response-cache requires --request-batch-size (e.g. 256)")
    if a.dp > 1 and "FLAG_DP_RANK" not in os.environ:
        from scripts.korbi.flag_eval.node_ep import launch

        launch(a.dp, list(sys.argv[1:] if argv is None else argv))
        return

    suite = {t["name"]: t for t in yaml.safe_load(open(a.suite))["tasks"]}

    def make_row(name):
        base, k, n = lmeval_parts.parse(name)
        t = suite.get(base)
        if t is None:
            raise ValueError(f"{name}: not in {a.suite}")
        if t["half"] != "vllm" or t["suite"] != "lm-eval-harness":
            raise ValueError(f"{name}: {t['half']} / {t['suite']} rows are not supported here (lm-eval on vLLM only)")
        if k is not None and a.limit:
            raise ValueError(f"{name}: parts and --limit exclude each other (lm-eval samples vs limit)")
        return {**t, "name": name, "part": None if k is None else (k, n)}

    try:
        rows = [make_row(name) for name in (a.names or [])][a.shard :: a.num_shards]
    except ValueError as exc:
        sys.exit(str(exc))
    export = Path(a.export).resolve()
    # test runs (--limit) never mark a full row done
    out = Path(a.out) / (export.name + (f"_limit{a.limit:g}" if a.limit else ""))
    done = out / "done"
    done.mkdir(parents=True, exist_ok=True)
    def _key(t):
        return (f"{t['task']}_{t['n_shot']}" if t["part"] is None
                else lmeval_parts.part_key(t["task"], t["n_shot"], *t["part"]))

    def is_done(t):  # a part is also done once its whole row is
        return (done / f"{_key(t)}.ok").exists() or (done / f"{t['task']}_{t['n_shot']}.ok").exists()

    rank0 = a.dp == 1 or int(os.environ["FLAG_DP_RANK"]) == 0
    if a.queue:
        from scripts.korbi.flag_eval.flag_queue import Queue

        queue = Queue(Path(a.queue))
        print(f"[flag_lmeval] {export.name} queue {a.queue}: {queue.status()}", flush=True)
        todo = None
    else:
        todo = [t for t in rows if not is_done(t)]
        if a.dp > 1:
            import torch.distributed as dist

            shared = [todo if rank0 else None]
            dist.broadcast_object_list(shared, src=0)
            todo = shared[0]
        print(f"[flag_lmeval] {export.name} shard {a.shard}/{a.num_shards}: {len(rows)} rows, {len(todo)} to run", flush=True)
        if not todo:
            return

    from lm_eval import simple_evaluate
    from lm_eval.loggers import EvaluationTracker
    from lm_eval.models.vllm_causallms import VLLM
    from lm_eval.tasks import TaskManager

    if a.dp > 1:
        from scripts.korbi.flag_eval.node_ep import distributed_adapter

        VLLM = distributed_adapter(VLLM)
        allow_empty_dp_shards()
    t0 = time.time()
    # The OELLM custom decode graphs are controlled by OELLM_DECODE_GRAPHS; this
    # vLLM engine flag stays opt-in because vLLM's own graph capture is separate.
    enforce_eager = os.environ.get("OELLM_VLLM_ENFORCE_EAGER", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    engine_extra = {}
    if a.max_num_batched_tokens is not None:
        engine_extra["max_num_batched_tokens"] = max(1, a.max_num_batched_tokens // a.dp)
    lm = VLLM(pretrained=str(export), trust_remote_code=True, dtype="bfloat16", batch_size=a.request_batch_size,
              tensor_parallel_size=a.tp, enable_expert_parallel=bool(a.ep) and (a.tp * a.dp > 1),
              gpu_memory_utilization=a.gpu_mem, max_num_seqs=max(1, a.max_num_seqs // a.dp), enforce_eager=enforce_eager,
              **engine_extra)
    if a.dp > 1:
        import torch

        lm._rank = int(os.environ["FLAG_DP_RANK"])
        lm._world_size = a.dp
        lm._device = torch.device("cpu")
    print(f"[flag_lmeval] engine up in {time.time() - t0:.0f} s", flush=True)
    tm = TaskManager(include_path=a.include_path)
    summary = out / f"summary_shard{a.shard}.tsv"
    cache_prefix = None
    if a.response_cache:
        import vllm
        import torch
        import lm_eval

        source = Path(vllm.__file__).parent
        identity = {
            "export": str(export), "tp": a.tp, "dp": a.dp, "ep": a.ep,
            "max_num_seqs": a.max_num_seqs, "request_batch": a.request_batch_size,
            "max_num_batched_tokens": a.max_num_batched_tokens,
            "vllm_version": vllm.__version__, "torch_version": torch.__version__,
            "harness_version": getattr(lm_eval, "__version__", "unknown"),
            "tokenizer": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in export.glob("*token*.json")},
            "files": [(f.name, f.stat().st_size, f.stat().st_mtime_ns)
                      for f in sorted(export.glob("*.safetensors"))],
            "config": hashlib.sha256((export / "config.json").read_bytes()).hexdigest(),
            "source": {str(f.relative_to(source)): hashlib.sha256(f.read_bytes()).hexdigest()
                       for f in [source / "model_executor/models/oellm_ambitious.py",
                                 source / "model_executor/layers/oellm_mhc.py",
                                 source / "model_executor/layers/oellm_fusion.py",
                                 source / "model_executor/layers/oellm_graph.py",
                                 source / "model_executor/layers/oellm_uva_table.py",
                                 source / "model_executor/layers/oellm_decode_kernels.py",
                                 source / "model_executor/layers/fused_moe/fused_moe.py",
                                 source / "model_executor/layers/fused_moe/activation.py"] if f.exists()},
            "settings": {k: os.environ.get(k) for k in ["OELLM_MOE", "OELLM_MHC_IMPL",
                         "OELLM_ENGRAM_TABLES", "OELLM_ENGRAM_LOOKUP", "OELLM_ENGRAM_MADVISE",
                         "OELLM_ENGRAM_DECODE_IMPL", "OELLM_FUSION_IMPL", "OELLM_DECODE_GRAPHS",
                         "OELLM_GRAPH_MAX_ROWS",
                         "OELLM_LOCAL_EXPERT_LOAD", "OELLM_ENGRAM_MODEL_DIR", "OELLM_DECODE_KERNELS",
                         "OELLM_DECODE_KERNELS_OFF", "OELLM_MOE_FP8", "OELLM_VLLM_CUDAGRAPH",
                         "OELLM_VLLM_ENFORCE_EAGER"]},
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        cache_dir = out / "response_cache" / digest
        cache_dir.mkdir(parents=True, exist_ok=True)
        if lm.rank == 0:
            (cache_dir / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
        if a.dp > 1:
            lm.barrier()
        cache_prefix = str(cache_dir / f"shard{a.shard}")
    failed = []

    def run_row(t) -> bool:
        """Evaluate one row or part; False if it failed (single engine; DP > 1 raises)."""
        key = _key(t)
        t1 = time.time()
        print(f"[flag_lmeval] --- {t['name']}", flush=True)
        try:
            if t["part"] is not None:
                if _run_part(t, a, lm, tm, out, done, cache_prefix, simple_evaluate, t1):
                    return True
                print(f"[flag_lmeval] {t['name']} part done in {time.time() - t1:.0f} s", flush=True)
                return True
            tracker = EvaluationTracker(output_path=str(out / "results" / key))
            res = simple_evaluate(model=lm, tasks=[t["task"]], num_fewshot=t["n_shot"], batch_size=a.request_batch_size,
                                  log_samples=True, evaluation_tracker=tracker, task_manager=tm,
                                  confirm_run_unsafe_code=True, limit=a.limit, use_cache=cache_prefix)
            if res is None:
                lm.barrier()
                return True
            samples = res.pop("samples")
            tracker.save_results_aggregated(results=res, samples=samples)
            for task_name in res["configs"]:
                tracker.save_results_samples(task_name=task_name, samples=samples[task_name])
            with summary.open("a") as f:
                for task_name, metrics in res["results"].items():
                    for m, v in metrics.items():
                        if isinstance(v, (int, float)) and m != "alias":
                            f.write(f"{t['name']}\t{task_name}\t{m}\t{v}\t{a.limit or ''}\n")
            (done / f"{key}.ok").write_text(json.dumps({"time_s": round(time.time() - t1, 1),
                                                       "limit": a.limit, "response_cache": cache_prefix,
                                                       "results": res["results"]}) + "\n")
            print(f"[flag_lmeval] {t['name']} done in {time.time() - t1:.0f} s: "
                  f"{json.dumps(res['results'].get(t['task'], {}), default=str)[:400]}", flush=True)
            if a.dp > 1:
                lm.barrier()
            return True
        except Exception:
            traceback.print_exc()
            if a.dp > 1:
                raise
            failed.append(t["name"])
            print(f"[flag_lmeval] {t['name']} FAILED after {time.time() - t1:.0f} s", flush=True)
            return False

    if todo is not None:
        for t in todo:
            run_row(t)
        n_run = len(todo)
    else:
        n_run = _queue_loop(queue, make_row, is_done, run_row, a.dp, rank0)
    print(f"[flag_lmeval] shard {a.shard}: {n_run - len(failed)} ok, failed: {failed}; {time.time() - t0:.0f} s", flush=True)
    if a.dp > 1:
        lm.model.llm_engine.engine_core.shutdown()
    sys.exit(1 if failed else 0)


def allow_empty_dp_shards() -> None:
    """A DP rank whose shard of a (sub)task has no documents contributes no requests instead of
    raising: lm-eval's build_all_requests raises "did not find any docs!" for an empty shard
    (include_base_44_estonian_health_oriented_education has 2 test documents for 4 ranks), the
    rank died and its peers waited in the next collective forever. The evaluator already pads
    the ranks' request counts and iterates no documents on an empty shard."""
    from lm_eval.api.task import Task

    if getattr(Task.build_all_requests, "_oellm_empty_ok", False):
        return
    original = Task.build_all_requests

    def build_all_requests(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except ValueError as exc:
            if "did not find any docs" not in str(exc) or kwargs.get("world_size", 1) <= 1:
                raise
            print(f"[flag_lmeval] {getattr(self, 'config', None) and self.config.task}: no documents on "
                  f"rank {kwargs.get('rank')} of {kwargs.get('world_size')}", flush=True)
            self._instances = []

    build_all_requests._oellm_empty_ok = True
    Task.build_all_requests = build_all_requests


def _queue_loop(queue, make_row, is_done, run_row, dp: int, rank0: bool) -> int:
    """Claim, evaluate and mark items until the queue is empty. With DP > 1 rank 0 claims and
    broadcasts each item (None: stop); its claim file is touched while the item runs."""
    import threading

    n = 0
    while True:
        claim = None
        if rank0:
            while True:
                claim = queue.claim()
                if claim is None:
                    break
                path, data = claim
                try:
                    t = make_row(data["item"])
                except ValueError as exc:
                    print(f"[flag_lmeval] queue item {data['item']}: {exc}", flush=True)
                    queue.fail(path, {**data, "attempts": queue.max_attempts})
                    continue
                if is_done(t):
                    queue.complete(path)
                    continue
                break
        name = claim[1]["item"] if claim else None
        if dp > 1:
            import torch.distributed as dist

            shared = [name]
            dist.broadcast_object_list(shared, src=0)
            name = shared[0]
        if name is None:
            return n
        stop = threading.Event()

        def heartbeat(path=claim[0] if claim else None):
            while not stop.wait(60):  # the claim's lease: other workers take it back if it ages
                queue.touch(path)

        if rank0:
            threading.Thread(target=heartbeat, daemon=True).start()
        ok = False
        try:
            ok = run_row(make_row(name))
        finally:
            stop.set()
            if rank0:
                (queue.complete(claim[0]) if ok else queue.fail(claim[0], claim[1]))
        n += 1


def _run_part(t, a, lm, tm, out, done, cache_prefix, simple_evaluate, t1) -> bool:
    """Part k of K of a row (lmeval_parts): evaluate its documents, keep their metric values, and
    merge the row if this part completes the set. True on the ranks without results (DP > 1)."""
    import fcntl

    k, n = t["part"]
    row_key = f"{t['task']}_{t['n_shot']}"
    res = simple_evaluate(model=lm, tasks=[t["task"]], num_fewshot=t["n_shot"], batch_size=a.request_batch_size,
                          log_samples=True, task_manager=tm, confirm_run_unsafe_code=True,
                          samples=lmeval_parts.part_samples(tm, t["task"], k, n), use_cache=cache_prefix)
    if res is None:
        lm.barrier()
        return True
    parts_dir = out / "parts" / row_key
    lmeval_parts.write_part(parts_dir / f"part{k}of{n}.json", t["task"], t["n_shot"], k, n, res.pop("samples"))
    (done / f"{lmeval_parts.part_key(t['task'], t['n_shot'], k, n)}.ok").write_text(
        json.dumps({"time_s": round(time.time() - t1, 1), "part": [k, n]}) + "\n")
    # the part that completes the set merges (a lock: two parts may finish at once)
    with open(parts_dir / ".merge.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        files = [parts_dir / f"part{i}of{n}.json" for i in range(n)]
        if all(f.exists() for f in files) and not (done / f"{row_key}.ok").exists():
            merged = lmeval_parts.merge(tm, t["task"], t["n_shot"], [json.loads(f.read_text()) for f in files])
            res_dir = out / "results" / row_key
            res_dir.mkdir(parents=True, exist_ok=True)
            (res_dir / f"results_merged_{n}parts.json").write_text(json.dumps(merged, indent=2, default=str) + "\n")
            (done / f"{row_key}.ok").write_text(json.dumps({"time_s": None, "parts": n, "limit": None,
                                                           "results": merged["results"]}, default=str) + "\n")
            print(f"[flag_lmeval] {t['task']}/{t['n_shot']} merged from {n} parts: "
                  f"{json.dumps(merged['results'].get(t['task'], {}), default=str)[:400]}", flush=True)
    if a.dp > 1:
        lm.barrier()
    return False


if __name__ == "__main__":
    main()
