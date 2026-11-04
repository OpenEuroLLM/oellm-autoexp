"""nsys capture of full-graph decode at one request per GPU (DP4/EP4): warmup, then a profiled
48-token generation (prefill + 47 graph-replayed decode steps)."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import time
import traceback

DP = 4


def worker(rank, args, port, barrier):
    os.environ.update(VLLM_DP_RANK=str(rank), VLLM_DP_RANK_LOCAL=str(rank), VLLM_DP_SIZE=str(DP),
                      VLLM_DP_MASTER_IP="127.0.0.1", VLLM_DP_MASTER_PORT=str(port))
    from vllm import LLM, SamplingParams
    prompts = json.loads(Path(args.prompts).read_text())
    llm = LLM(model=args.model, trust_remote_code=True, dtype="bfloat16", tensor_parallel_size=1,
              enable_expert_parallel=True, all2all_backend="allgather_reducescatter",
              gpu_memory_utilization=0.85, max_num_seqs=64, max_num_batched_tokens=8192,
              max_model_len=4096, enable_prefix_caching=False, seed=42, disable_log_stats=False,
              enforce_eager=False, compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                                                       "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64]},
              worker_extension_cls="scripts.korbi.vllm.cuda_profiler_probe.CudaProfilerProbe")
    work = [{"prompt_token_ids": prompts[rank * 64][:args.prompt_len]}]
    sp = SamplingParams(temperature=0, max_tokens=48, ignore_eos=True)
    for _ in range(2):
        barrier.wait(timeout=3600)
        llm.generate(work, sp, use_tqdm=False)
    barrier.wait(timeout=3600)
    llm.collective_rpc("prof_start")
    t0 = time.perf_counter()
    out = llm.generate(work, sp, use_tqdm=False)
    wall = time.perf_counter() - t0
    llm.collective_rpc("prof_stop")
    m = out[0].metrics
    print(json.dumps({"rank": rank, "wall": wall, "decode_tok_s": 47 / (m.last_token_ts - m.first_token_ts)}), flush=True)
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
    p.add_argument("--prompt-len", type=int, default=256)
    a = p.parse_args()
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
                raise RuntimeError("profile worker failed")
            time.sleep(0.5)
    finally:
        for pr in procs:
            if pr.is_alive():
                pr.terminate()
        for pr in procs:
            pr.join(timeout=5)


if __name__ == "__main__":
    main()
