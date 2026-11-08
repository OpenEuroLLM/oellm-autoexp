"""Warm latency controls and SDK-triggered decode-only Nsight captures."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import time
import traceback


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def worker(rank, dp, tp, port, args, barrier):
    if dp > 1:
        os.environ.update(VLLM_DP_RANK=str(rank), VLLM_DP_RANK_LOCAL=str(rank),
                          VLLM_DP_SIZE=str(dp), VLLM_DP_MASTER_IP='127.0.0.1',
                          VLLM_DP_MASTER_PORT=str(port))
    from vllm import LLM, SamplingParams
    prompts = json.loads(Path(args.prompts).read_text())
    result = {'rank': rank, 'layout': args.layout, 'cases': []}
    started = time.perf_counter()
    llm = LLM(model=args.model, trust_remote_code=True, dtype='bfloat16',
        tensor_parallel_size=tp, enable_expert_parallel=True,
        all2all_backend='allgather_reducescatter', gpu_memory_utilization=0.85,
        max_num_seqs=256 // dp, max_num_batched_tokens=32768 // dp,
        max_model_len=4096, enforce_eager=True, enable_prefix_caching=False,
        disable_log_stats=False, seed=42,
        worker_extension_cls='scripts.korbi.vllm.gpu_profile.DecodeTimelineProbe')
    result['startup_seconds'] = time.perf_counter() - started
    result['workers'] = llm.collective_rpc('batching_memory', timeout=30)
    for batch in [1, 32, 256]:
        chosen = prompts[:batch][rank::dp]
        dummy = not chosen
        work = [{'prompt_token_ids': p} for p in (chosen if chosen else prompts[:1])]
        entry = {'batch': batch, 'real_requests': len(chosen), 'dummy_start_requests': int(dummy)}
        barrier.wait(timeout=1800)
        llm.generate(work, SamplingParams(temperature=0, max_tokens=1 if dummy else 32,
                                          ignore_eos=True), use_tqdm=False)
        barrier.wait(timeout=1800)
        begin = time.perf_counter()
        outputs = llm.generate(work, SamplingParams(temperature=0, max_tokens=1 if dummy else 128,
                                                   ignore_eos=True), use_tqdm=False)
        entry['unprofiled_wall_seconds'] = time.perf_counter() - begin
        entry['timings'] = []
        if not dummy:
            for output in outputs:
                assert len(output.outputs[0].token_ids) == 128
                m = output.metrics
                assert m is not None and m.last_token_ts > m.first_token_ts > 0
                entry['timings'].append({'first_token_ts': m.first_token_ts,
                    'last_token_ts': m.last_token_ts, 'first_token_latency': m.first_token_latency})
        label = f'{args.layout}_b{batch}'
        barrier.wait(timeout=1800)
        entry['arm'] = llm.collective_rpc('arm_gpu_trace', timeout=30,
                                        kwargs={'label': label, 'skip': 8, 'steps': 12})
        barrier.wait(timeout=1800)
        llm.generate(work, SamplingParams(temperature=0, max_tokens=1 if dummy else 64,
                                          ignore_eos=True), use_tqdm=False)
        barrier.wait(timeout=1800)
        entry['disarm'] = llm.collective_rpc('disarm_gpu_trace', timeout=60)
        for item in entry['disarm']:
            if item['controller']:
                assert item['capture_completed'], item
        result['cases'].append(entry)
        save(Path(args.out) / f'rank{rank}.json', result)
        print(json.dumps({'layout': args.layout, 'rank': rank, 'batch': batch,
                          'unprofiled_wall_seconds': entry['unprofiled_wall_seconds']}), flush=True)
    barrier.wait(timeout=1800)
    llm.llm_engine.engine_core.shutdown()


def entry(*args):
    try:
        worker(*args)
    except BaseException:
        print(traceback.format_exc(), flush=True)
        os._exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layout', choices=['ep4', 'tp4'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--prompts', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    dp, tp = (4, 1) if args.layout == 'ep4' else (1, 4)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    ctx = mp.get_context('spawn')
    barrier = ctx.Barrier(dp)
    procs = [ctx.Process(target=entry, args=(rank, dp, tp, port, args, barrier)) for rank in range(dp)]
    for proc in procs:
        proc.start()
    try:
        while any(proc.is_alive() for proc in procs):
            bad = [(proc.pid, proc.exitcode) for proc in procs if proc.exitcode not in (None, 0)]
            if bad:
                raise RuntimeError(f'Worker failure: {bad}')
            time.sleep(0.5)
        for proc in procs:
            proc.join()
        assert all(proc.exitcode == 0 for proc in procs)
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(timeout=5)


if __name__ == '__main__':
    main()
