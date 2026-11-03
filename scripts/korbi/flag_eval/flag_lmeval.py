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
per (row, metric)."""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--names", nargs="+", required=True, help="flag_evals.yaml names half/task/n_shot")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--tp", type=int, default=4)
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
    a = p.parse_args()
    if a.response_cache and a.request_batch_size == "auto":
        p.error("--response-cache requires --request-batch-size (e.g. 256)")

    suite = {t["name"]: t for t in yaml.safe_load(open(a.suite))["tasks"]}
    rows = []
    for name in a.names:
        t = suite.get(name)
        if t is None:
            sys.exit(f"{name}: not in {a.suite}")
        if t["half"] != "vllm" or t["suite"] != "lm-eval-harness":
            sys.exit(f"{name}: {t['half']} / {t['suite']} rows are not supported here (lm-eval on vLLM only)")
        rows.append(t)
    rows = rows[a.shard :: a.num_shards]
    export = Path(a.export).resolve()
    # test runs (--limit) never mark a full row done
    out = Path(a.out) / (export.name + (f"_limit{a.limit:g}" if a.limit else ""))
    done = out / "done"
    done.mkdir(parents=True, exist_ok=True)
    todo = [t for t in rows if not (done / f"{t['task']}_{t['n_shot']}.ok").exists()]
    print(f"[flag_lmeval] {export.name} shard {a.shard}/{a.num_shards}: {len(rows)} rows, {len(todo)} to run", flush=True)
    if not todo:
        return

    from lm_eval import simple_evaluate
    from lm_eval.loggers import EvaluationTracker
    from lm_eval.models.vllm_causallms import VLLM
    from lm_eval.tasks import TaskManager

    t0 = time.time()
    engine_extra = {}
    if a.max_num_batched_tokens is not None:
        engine_extra["max_num_batched_tokens"] = a.max_num_batched_tokens
    lm = VLLM(pretrained=str(export), trust_remote_code=True, dtype="bfloat16", batch_size=a.request_batch_size,
              tensor_parallel_size=a.tp, enable_expert_parallel=bool(a.ep) and a.tp > 1,
              gpu_memory_utilization=a.gpu_mem, max_num_seqs=a.max_num_seqs, enforce_eager=True,
              **engine_extra)
    print(f"[flag_lmeval] engine up in {time.time() - t0:.0f} s", flush=True)
    tm = TaskManager(include_path=a.include_path)
    summary = out / f"summary_shard{a.shard}.tsv"
    cache_prefix = None
    if a.response_cache:
        import vllm

        source = Path(vllm.__file__).parent
        identity = {
            "export": str(export), "tp": a.tp, "ep": a.ep,
            "max_num_seqs": a.max_num_seqs, "request_batch": a.request_batch_size,
            "max_num_batched_tokens": a.max_num_batched_tokens,
            "vllm_version": vllm.__version__,
            "files": [(f.name, f.stat().st_size, f.stat().st_mtime_ns)
                      for f in sorted(export.glob("*.safetensors"))],
            "config": hashlib.sha256((export / "config.json").read_bytes()).hexdigest(),
            "source": {str(f.relative_to(source)): hashlib.sha256(f.read_bytes()).hexdigest()
                       for f in [source / "model_executor/models/oellm_ambitious.py",
                                 source / "model_executor/layers/oellm_mhc.py"] if f.exists()},
            "settings": {k: os.environ.get(k) for k in ["OELLM_MOE", "OELLM_MHC_IMPL",
                         "OELLM_ENGRAM_TABLES", "OELLM_ENGRAM_LOOKUP", "OELLM_ENGRAM_MADVISE",
                         "OELLM_LOCAL_EXPERT_LOAD", "OELLM_ENGRAM_MODEL_DIR"]},
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        cache_dir = out / "response_cache" / digest
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
        cache_prefix = str(cache_dir / f"shard{a.shard}")
    failed = []
    for t in todo:
        key = f"{t['task']}_{t['n_shot']}"
        t1 = time.time()
        print(f"[flag_lmeval] --- {t['name']}", flush=True)
        try:
            tracker = EvaluationTracker(output_path=str(out / "results" / key))
            res = simple_evaluate(model=lm, tasks=[t["task"]], num_fewshot=t["n_shot"], batch_size=a.request_batch_size,
                                  log_samples=True, evaluation_tracker=tracker, task_manager=tm,
                                  confirm_run_unsafe_code=True, limit=a.limit, use_cache=cache_prefix)
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
        except Exception:
            traceback.print_exc()
            failed.append(t["name"])
            print(f"[flag_lmeval] {t['name']} FAILED after {time.time() - t1:.0f} s", flush=True)
    print(f"[flag_lmeval] shard {a.shard}: {len(todo) - len(failed)} ok, failed: {failed}; {time.time() - t0:.0f} s", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
