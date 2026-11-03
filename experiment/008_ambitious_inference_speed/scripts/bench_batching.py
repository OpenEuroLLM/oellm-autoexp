"""Saturated 1024-request concurrency sweep; independently on each four-GPU node."""
import argparse
import functools
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback


class SchedulerPeaks:
    def __init__(self):
        self.reset()

    def reset(self):
        self.engines = {}

    def record(self, scheduler_stats, iteration_stats, **kwargs):
        engine = str(kwargs.get('engine_idx', 0))
        item = self.engines.setdefault(engine, {'running': 0, 'waiting': 0,
                                              'cache_usage': 0.0, 'preemptions': 0})
        if scheduler_stats is not None:
            item['running'] = max(item['running'], scheduler_stats.num_running_reqs)
            item['waiting'] = max(item['waiting'], scheduler_stats.num_waiting_reqs)
            item['cache_usage'] = max(item['cache_usage'], scheduler_stats.kv_cache_usage)
        if iteration_stats is not None:
            item['preemptions'] += getattr(iteration_stats, 'num_preemptions', 0)

    def log(self):
        pass


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def worker(rank, dp, tp, port, args, barrier):
    if dp > 1:
        os.environ.update(VLLM_DP_RANK=str(rank), VLLM_DP_RANK_LOCAL=str(rank),
                          VLLM_DP_SIZE=str(dp), VLLM_DP_MASTER_IP='127.0.0.1',
                          VLLM_DP_MASTER_PORT=str(port))
    import torch
    from vllm import LLM, SamplingParams
    prompts = json.loads(Path(args.prompts_file).read_text())[rank::dp]
    records = [{'prompt_token_ids': ids} for ids in prompts]
    out = Path(args.out)
    result = {'rank': rank, 'tp': tp, 'dp': dp, 'trials': []}
    started = time.perf_counter()
    llm = LLM(model=args.model, trust_remote_code=True, dtype='bfloat16',
              tensor_parallel_size=tp, enable_expert_parallel=True,
              all2all_backend='allgather_reducescatter', gpu_memory_utilization=0.85,
              max_num_seqs=args.max_seqs // dp, max_model_len=4096,
              max_num_batched_tokens=args.token_budget // dp,
              enforce_eager=True, enable_prefix_caching=False,
              disable_log_stats=False, seed=42,
              worker_extension_cls='scripts.korbi.vllm.batch_memory.BatchMemoryProbe')
    result['startup_seconds'] = time.perf_counter() - started
    atomic_json(out / f'rank{rank}.json', result)
    peaks = SchedulerPeaks()
    assert llm.llm_engine.logger_manager is not None
    llm.llm_engine.logger_manager.stat_loggers.append(peaks)
    result['memory_after_init'] = llm.collective_rpc('batching_memory', timeout=30)
    atomic_json(out / f'rank{rank}.json', result)
    if args.engram_compare:
        result['engram_checks'] = llm.collective_rpc('check_batched_engram', timeout=300)
        atomic_json(out / f'rank{rank}.json', result)
    for variant in (['reference', 'batched'] if args.engram_compare else ['baseline']):
        if args.engram_compare:
            configured = llm.collective_rpc('configure_engram_decode', timeout=30,
                                            kwargs={'batched': variant == 'batched'})
            assert all(count > 0 for count in configured)
        for mode in ['scoring', 'generation']:
            params = SamplingParams(temperature=0, max_tokens=1 if mode == 'scoring' else args.decode,
                                    prompt_logprobs=0 if mode == 'scoring' else None,
                                    ignore_eos=True)
            barrier.wait(timeout=1800)
            # Warm full decode shapes on up to two complete concurrency waves.
            begin = time.perf_counter()
            warm = llm.generate(records[:2 * args.max_seqs // dp], params, use_tqdm=False)
            assert all(len(x.outputs[0].token_ids) == params.max_tokens for x in warm)
            result[f'{mode}_warmup_seconds'] = time.perf_counter() - begin
            del warm
            repeats = 2 if mode == 'scoring' else 1
            for trial in range(repeats):
                barrier.wait(timeout=1800)
                peaks.reset()
                llm.collective_rpc('batching_memory', timeout=30, kwargs={'reset': True})
                begin = time.perf_counter()
                outputs = llm.generate(records, params, use_tqdm=False)
                elapsed = time.perf_counter() - begin
                assert len(outputs) == len(records)
                assert all(len(x.outputs[0].token_ids) == params.max_tokens for x in outputs)
                entry = {'variant': variant, 'mode': mode, 'trial': trial, 'seconds': elapsed,
                         'requests': len(outputs), 'input_tokens': sum(map(len, prompts)),
                         'output_tokens': sum(len(x.outputs[0].token_ids) for x in outputs),
                         'scheduler_peaks': peaks.engines,
                         'worker_memory': llm.collective_rpc('batching_memory', timeout=30)}
                if mode == 'scoring':
                    values = []
                    sentinels = []
                    for i, (ids, output) in enumerate(zip(prompts, outputs, strict=True)):
                        lp = [part[ids[j]].logprob for j, part in
                              enumerate(output.prompt_logprobs) if part is not None]
                        assert len(lp) == len(ids) - 1
                        values.extend(lp)
                        if i < 8:
                            sentinels.append({'index': rank + i * dp, 'logprobs': lp})
                    entry.update(logprob_sum=sum(values), scored_tokens=len(values), sentinels=sentinels)
                else:
                    entry['generated'] = [{'index': rank + i * dp, 'tokens': x.outputs[0].token_ids}
                                          for i, x in enumerate(outputs)]
                result['trials'].append(entry)
                atomic_json(out / f'rank{rank}.json', result)
                print(json.dumps({'rank': rank, 'mode': mode, 'trial': trial,
                                  'seconds': elapsed, 'max_seqs': args.max_seqs,
                                  'token_budget': args.token_budget}), flush=True)
                del outputs
        if args.engram_compare:
            from decode_diagnostics import measure_decode
            scaling = measure_decode(llm, rank, dp, records, args, barrier,
                                     batches=[1, 32, 256], profile=False, label=variant)
            result.setdefault('decode_scaling_by_variant', {})[variant] = scaling
            atomic_json(out / f'rank{rank}.json', result)
    if not args.engram_compare and args.max_seqs == 256 and args.token_budget == 32768:
        from decode_diagnostics import measure_decode
        result['decode_diagnostics'] = measure_decode(
            llm, rank, dp, records, args, barrier)
        atomic_json(out / f'rank{rank}.json', result)
    barrier.wait(timeout=1800)
    llm.llm_engine.engine_core.shutdown()


def worker_entry(rank, dp, tp, port, args, barrier):
    try:
        worker(rank, dp, tp, port, args, barrier)
    except BaseException:
        error = traceback.format_exc()
        atomic_json(Path(args.out) / f'error_rank{rank}.json', {'traceback': error})
        print(error, file=sys.stderr, flush=True)
        # multiprocessing's finalizer otherwise joins engine children before
        # printing the exception. The case supervisor cleans the process group.
        os._exit(1)


def run_case(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dp, tp = (4, 1) if args.layout == 'ep4' else (1, 4)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    ctx = mp.get_context('spawn')
    barrier = ctx.Barrier(dp)
    processes = [ctx.Process(target=worker_entry, args=(r, dp, tp, port, args, barrier))
                 for r in range(dp)]
    for proc in processes:
        proc.start()
    try:
        while any(proc.is_alive() for proc in processes):
            bad = [(proc.pid, proc.exitcode) for proc in processes
                   if proc.exitcode not in (None, 0)]
            if bad:
                raise RuntimeError(f'worker failures: {bad}')
            time.sleep(0.5)
        for proc in processes:
            proc.join()
        if any(proc.exitcode for proc in processes):
            raise RuntimeError('worker failed')
    finally:
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
        for proc in processes:
            proc.join(timeout=5)
    ranks = [json.loads((out / f'rank{r}.json').read_text()) for r in range(dp)]
    result = {'layout': args.layout, 'max_seqs': args.max_seqs,
              'token_budget': args.token_budget, 'dp': dp, 'tp': tp,
              'startup_seconds': max(r['startup_seconds'] for r in ranks),
              'prompt_sha256': hashlib.sha256(Path(args.prompts_file).read_bytes()).hexdigest(),
              'trials': []}
    for i in range(len(ranks[0]['trials'])):
        trials = [r['trials'][i] for r in ranks]
        elapsed = max(t['seconds'] for t in trials)
        entry = {key: sum(t[key] for t in trials)
                 for key in ['requests', 'input_tokens', 'output_tokens']}
        entry.update(variant=trials[0].get('variant', 'baseline'),
                     mode=trials[0]['mode'], trial=trials[0]['trial'], seconds=elapsed,
                     requests_per_second=entry['requests'] / elapsed,
                     input_tokens_per_second=entry['input_tokens'] / elapsed,
                     output_tokens_per_second=entry['output_tokens'] / elapsed)
        if entry['mode'] == 'scoring':
            entry['mean_nll'] = -sum(t['logprob_sum'] for t in trials) / sum(t['scored_tokens'] for t in trials)
        entry['scheduler_peaks_by_frontend'] = [t['scheduler_peaks'] for t in trials]
        entry['worker_memory_by_frontend'] = [t['worker_memory'] for t in trials]
        result['trials'].append(entry)
    if all('decode_diagnostics' in r for r in ranks):
        from decode_diagnostics import combine
        result['decode_scaling'] = combine([r['decode_diagnostics'] for r in ranks])
        atomic_json(out / 'decode_scaling.json', result['decode_scaling'])
    if args.engram_compare:
        from decode_diagnostics import combine
        result['engram_checks_by_rank'] = [r['engram_checks'] for r in ranks]
        result['decode_scaling_by_variant'] = {
            variant: combine([r['decode_scaling_by_variant'][variant] for r in ranks])
            for variant in ['reference', 'batched']}
    atomic_json(out / 'summary.json', result)
    print(json.dumps(result), flush=True)


def memory_sampler(stop, rows):
    while not stop.is_set():
        try:
            sample = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'], capture_output=True,
                                    text=True, timeout=10, check=True)
            rows.append({'time': time.time(), 'gpus': sample.stdout.strip()})
        except Exception as error:
            rows.append({'time': time.time(), 'error': str(error)})
        stop.wait(2)


def run_sweep(args):
    import oellm_autoexp.hf_export.score_on_flag as reader
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts_file = out / 'prompts.json'
    if not prompts_file.exists():
        mix = out / 'datamix.txt'
        mix.write_text('1.0 /gpfs/projects/ehpc533/users/korbip/data/v8ml-valid/all_text_document\n')
        reader.read_index = functools.lru_cache(maxsize=None)(reader.read_index)
        docs = reader.sample_docs(mix, args.requests, 512, 0)
        atomic_json(prompts_file, [[1] + d[:255].tolist() for d in docs])
    cases = ([(256, 32768)] if args.engram_compare else
             [(256, 65536)] if args.extra_only else
             [(32, 8192)] + [(size, budget) for budget in [8192, 32768]
                             for size in [64, 128, 256]])
    status_path = out / ('candidate_status.json' if args.engram_compare else
                         'status_extra.json' if args.extra_only else 'status.json')
    status = []
    for size, budget in cases:
        case_out = out / f's{size}_t{budget}'
        if (case_out / 'summary.json').exists():
            status.append({'max_seqs': size, 'token_budget': budget, 'status': 'previously_completed'})
            continue
        end = int(os.environ.get('SLURM_JOB_END_TIME', '0'))
        remaining = end - time.time() if end else args.case_timeout
        if remaining < 210:
            status.append({'max_seqs': size, 'token_budget': budget, 'status': 'insufficient_allocation_time'})
            atomic_json(status_path, status)
            continue
        case_out.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, __file__, '--case', '--layout', args.layout,
                   '--model', args.model, '--out', str(case_out), '--max-seqs', str(size),
                   '--token-budget', str(budget), '--prompts-file', str(prompts_file),
                   '--decode', str(args.decode)]
        if args.engram_compare:
            command.append('--engram-compare')
        stop, samples = threading.Event(), []
        monitor = threading.Thread(target=memory_sampler, args=(stop, samples), daemon=True)
        monitor.start()
        started = time.time()
        with (case_out / 'run.log').open('w') as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            timed_out = False
            try:
                code = proc.wait(timeout=min(args.case_timeout, remaining - 30))
            except subprocess.TimeoutExpired:
                code, timed_out = None, True
            finally:
                # Clean the complete case process group, including EP engine descendants.
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    time.sleep(2)
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=20)
                stop.set()
                monitor.join(timeout=12)
        atomic_json(case_out / 'gpu_samples.json', samples)
        item = {'max_seqs': size, 'token_budget': budget, 'exit_code': code,
                'status': 'timeout' if timed_out else 'ok' if code == 0 else 'failed',
                'wall_seconds': time.time() - started}
        status.append(item)
        atomic_json(status_path, status)
        print(json.dumps(item), flush=True)
    if any(x['status'] not in ['ok', 'previously_completed'] for x in status):
        sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layout', required=True, choices=['tp4', 'ep4'])
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--requests', type=int, default=1024)
    p.add_argument('--decode', type=int, default=32)
    p.add_argument('--case-timeout', type=int, default=1500)
    p.add_argument('--case', action='store_true')
    p.add_argument('--extra-only', action='store_true')
    p.add_argument('--engram-compare', action='store_true')
    p.add_argument('--max-seqs', type=int)
    p.add_argument('--token-budget', type=int)
    p.add_argument('--prompts-file')
    a = p.parse_args()
    try:
        run_case(a) if a.case else run_sweep(a)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
