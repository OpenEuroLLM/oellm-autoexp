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
    p.add_argument("--names", nargs="+", required=True, help="flag_evals.yaml names half/task/n_shot")
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
    rows = []
    for name in a.names:
        base, k, n = lmeval_parts.parse(name)
        t = suite.get(base)
        if t is None:
            sys.exit(f"{name}: not in {a.suite}")
        if t["half"] != "vllm" or t["suite"] != "lm-eval-harness":
            sys.exit(f"{name}: {t['half']} / {t['suite']} rows are not supported here (lm-eval on vLLM only)")
        if k is not None and a.limit:
            sys.exit(f"{name}: parts and --limit exclude each other (lm-eval samples vs limit)")
        rows.append({**t, "name": name, "part": None if k is None else (k, n)})
    rows = rows[a.shard :: a.num_shards]
    export = Path(a.export).resolve()
    # test runs (--limit) never mark a full row done
    out = Path(a.out) / (export.name + (f"_limit{a.limit:g}" if a.limit else ""))
    done = out / "done"
    done.mkdir(parents=True, exist_ok=True)
    def _key(t):
        return (f"{t['task']}_{t['n_shot']}" if t["part"] is None
                else lmeval_parts.part_key(t["task"], t["n_shot"], *t["part"]))

    # a part is also done once its whole row is
    todo = [t for t in rows if not (done / f"{_key(t)}.ok").exists()
            and not (done / f"{t['task']}_{t['n_shot']}.ok").exists()]
    if a.dp > 1:
        import torch.distributed as dist

        shared = [todo if int(os.environ["FLAG_DP_RANK"]) == 0 else None]
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
    for t in todo:
        key = _key(t)
        t1 = time.time()
        print(f"[flag_lmeval] --- {t['name']}", flush=True)
        try:
            if t["part"] is not None:
                if _run_part(t, a, lm, tm, out, done, cache_prefix, simple_evaluate, t1):
                    continue
                print(f"[flag_lmeval] {t['name']} part done in {time.time() - t1:.0f} s", flush=True)
                continue
            tracker = EvaluationTracker(output_path=str(out / "results" / key))
            res = simple_evaluate(model=lm, tasks=[t["task"]], num_fewshot=t["n_shot"], batch_size=a.request_batch_size,
                                  log_samples=True, evaluation_tracker=tracker, task_manager=tm,
                                  confirm_run_unsafe_code=True, limit=a.limit, use_cache=cache_prefix)
            if res is None:
                lm.barrier()
                continue
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
        except Exception:
            traceback.print_exc()
            if a.dp > 1:
                raise
            failed.append(t["name"])
            print(f"[flag_lmeval] {t['name']} FAILED after {time.time() - t1:.0f} s", flush=True)
    print(f"[flag_lmeval] shard {a.shard}: {len(todo) - len(failed)} ok, failed: {failed}; {time.time() - t0:.0f} s", flush=True)
    if a.dp > 1:
        lm.model.llm_engine.engine_core.shutdown()
    sys.exit(1 if failed else 0)


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
