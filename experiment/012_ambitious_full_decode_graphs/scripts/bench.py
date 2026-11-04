"""Single-request / per-GPU-batch decode speed of one vLLM engine layout (TP1/DP4/EP4).

Variants (--variant): islands = experiment 010's best (eager, fused mHC/norms, local graph
islands, RAM Engram tables); fullgraph = vLLM's FULL_DECODE_ONLY CUDA graph over the whole
decode step (no torch.compile), Engram tables pinned in host memory (OELLM_ENGRAM_TABLES=uva).
Decode rates come from the engine's first/last-token timestamps (prefill excluded)."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import time
import traceback

DP = 4


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def worker(rank, args, port, barrier):
    os.environ.update(VLLM_DP_RANK=str(rank), VLLM_DP_RANK_LOCAL=str(rank), VLLM_DP_SIZE=str(DP),
                      VLLM_DP_MASTER_IP="127.0.0.1", VLLM_DP_MASTER_PORT=str(port))
    from vllm import LLM, SamplingParams
    prompts = json.loads(Path(args.prompts).read_text())
    out = Path(args.out) / f"{args.variant}_rank{rank}.json"
    kw = dict(model=args.model, trust_remote_code=True, dtype="bfloat16", tensor_parallel_size=1,
              enable_expert_parallel=True, all2all_backend="allgather_reducescatter",
              gpu_memory_utilization=0.85, max_num_seqs=64, max_num_batched_tokens=8192,
              max_model_len=4096, enable_prefix_caching=False, seed=42,
              disable_log_stats=False)
    if args.variant == "fullgraph":
        kw.update(enforce_eager=False, compilation_config={
            "mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64]})
    else:
        kw.update(enforce_eager=True)
    started = time.perf_counter()
    llm = LLM(**kw)
    res = {"rank": rank, "variant": args.variant, "startup_seconds": time.perf_counter() - started,
           "env": {k: v for k, v in os.environ.items() if k.startswith("OELLM_")}, "trials": []}
    save(out, res)
    # (label, requests on this rank): one request on the whole node, then 1 / 8 / 32 per GPU
    cases = [("node1", 1 if rank == 0 else 0), ("gpu1", 1), ("gpu8", 8), ("gpu32", 32)]
    for label, n in cases:
        mine = prompts[rank * 64: rank * 64 + n]
        work = [{"prompt_token_ids": p} for p in (mine or prompts[:1])]
        for phase in ("warmup", "timed"):
            barrier.wait(timeout=3600)
            tokens = (32 if phase == "warmup" else args.tokens) if mine else 1
            t0 = time.perf_counter()
            outs = llm.generate(work, SamplingParams(temperature=0, max_tokens=tokens, ignore_eos=True),
                                use_tqdm=False)
            wall = time.perf_counter() - t0
        entry = {"case": label, "requests": len(mine), "wall_seconds": wall, "per_request": []}
        for o in (outs if mine else []):
            m = o.metrics
            k = len(o.outputs[0].token_ids)
            entry["per_request"].append({"tokens": o.outputs[0].token_ids[:16],
                                         "decode_tok_s": (k - 1) / (m.last_token_ts - m.first_token_ts)})
        if entry["per_request"]:
            entry["mean_request_decode_tok_s"] = sum(r["decode_tok_s"] for r in entry["per_request"]) / len(entry["per_request"])
        res["trials"].append(entry)
        save(out, res)
        print(json.dumps({"variant": args.variant, "rank": rank, "case": label,
                          "decode_tok_s": entry.get("mean_request_decode_tok_s"), "wall": wall}), flush=True)
    barrier.wait(timeout=3600)
    llm.llm_engine.engine_core.shutdown()


def entry(*a):
    try:
        worker(*a)
    except BaseException:
        print(traceback.format_exc(), flush=True)
        os._exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--variant", choices=["islands", "fullgraph"], required=True)
    p.add_argument("--tokens", type=int, default=256)
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(DP)
    procs = [ctx.Process(target=entry, args=(r, a, port, barrier)) for r in range(DP)]
    for pr in procs:
        pr.start()
    try:
        while any(pr.is_alive() for pr in procs):
            if any(pr.exitcode not in (None, 0) for pr in procs):
                raise RuntimeError("benchmark worker failed")
            time.sleep(0.5)
    finally:
        for pr in procs:
            if pr.is_alive():
                pr.terminate()
        for pr in procs:
            pr.join(timeout=5)


if __name__ == "__main__":
    main()
