"""Single-request and batched decode timing, followed by untimed Python profiles."""
import json
import os
from pathlib import Path
import subprocess
import time


def save(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def measure_decode(llm, rank, dp, records, args, barrier,
                   batches=None, profile=True, label=""):
    from vllm import SamplingParams
    out = Path(args.out)
    data = {'rank': rank, 'variant': label, 'measurements': [], 'profiles': []}
    prefix = label + '_' if label else ''
    data_path = out / f'diagnostic_{prefix}rank{rank}.json'
    for batch in (batches if batches is not None else [1, 8, 32, 64, 128, 256]):
        chosen = [records[i // dp] for i in range(rank, batch, dp)]
        # Start every DP frontend. After the first step only the real request
        # remains in the batch-one decode interval; idle ranks run dummy steps.
        dummy = not chosen
        work = chosen if chosen else records[:1]
        for tokens in [8, 128]:
            params = SamplingParams(temperature=0, max_tokens=1 if dummy else tokens,
                                    ignore_eos=True)
            barrier.wait(timeout=1800)
            begin = time.perf_counter()
            outputs = llm.generate(work, params, use_tqdm=False)
            seconds = time.perf_counter() - begin
            if dummy:
                outputs = []
            if tokens == 8:
                continue
            timings = []
            for output in outputs:
                assert len(output.outputs[0].token_ids) == tokens
                m = output.metrics
                assert m is not None and m.last_token_ts > m.first_token_ts > 0
                timings.append({'first_token_ts': m.first_token_ts,
                                'last_token_ts': m.last_token_ts,
                                'first_token_latency': m.first_token_latency,
                                'decode_seconds': m.last_token_ts - m.first_token_ts,
                                'output_tokens': tokens})
            entry = {'batch': batch, 'requests': len(chosen), 'seconds': seconds,
                     'tokens_per_request': tokens, 'request_timings': timings,
                     'dummy_start_requests': int(dummy)}
            data['measurements'].append(entry)
            save(data_path, data)
            print(json.dumps({'diagnostic': 'decode_scaling', 'rank': rank,
                              'batch': batch, 'seconds': seconds}), flush=True)
    if not profile:
        return data
    # Profiling is a separate workload, excluded from every throughput measurement.
    spy = '/gpfs/projects/ehpc533/users/korbip/hybrid_v019/tools/py-spy'
    workers = llm.collective_rpc('batching_memory', timeout=30)
    targets = [os.getpid()] + [m['pid'] for m in workers]
    for batch in [32, 256]:
        barrier.wait(timeout=1800)
        spies = []
        for pid in targets:
            path = out / f'profile_b{batch}_rank{rank}_pid{pid}.json'
            log = path.with_suffix('.txt').open('w')
            process = subprocess.Popen([spy, 'record', '--pid', str(pid), '--rate', '49',
                                        '--duration', '8', '--idle', '--nonblocking',
                                        '--format', 'speedscope', '--output', str(path)],
                                       stdout=log, stderr=subprocess.STDOUT)
            spies.append((process, log, path, pid))
        work = [records[i // dp] for i in range(rank, batch, dp)]
        outputs = llm.generate(work, SamplingParams(temperature=0, max_tokens=64,
                                                   ignore_eos=True), use_tqdm=False)
        assert all(len(x.outputs[0].token_ids) == 64 for x in outputs)
        for process, log, path, pid in spies:
            try:
                code = process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
            log.close()
            data['profiles'].append({'batch': batch, 'pid': pid, 'path': path.name,
                                     'exit_code': code})
        save(data_path, data)
    return data


def combine(ranks):
    result = []
    for index in range(len(ranks[0]['measurements'])):
        items = [r['measurements'][index] for r in ranks]
        timings = [t for item in items for t in item['request_timings']]
        count = sum(item['requests'] for item in items)
        tokens = items[0]['tokens_per_request']
        assert count == items[0]['batch']
        span = max(t['last_token_ts'] for t in timings) - min(t['first_token_ts'] for t in timings)
        decode_rates = [(tokens - 1) / t['decode_seconds'] for t in timings]
        result.append({'batch': count, 'tokens_per_request': tokens,
                       'wall_seconds': max(item['seconds'] for item in items),
                       'end_to_end_output_tokens_per_second': count * tokens / max(item['seconds'] for item in items),
                       'decode_span_seconds': span,
                       'decode_output_tokens_per_second': count * (tokens - 1) / span,
                       'mean_per_request_decode_tokens_per_second': sum(decode_rates) / count,
                       'mean_time_to_first_token_seconds': sum(t['first_token_latency'] for t in timings) / count,
                       'dummy_start_requests': sum(item['dummy_start_requests'] for item in items)})
    single = result[0]['decode_output_tokens_per_second']
    for row in result:
        row['decode_speedup_vs_single'] = row['decode_output_tokens_per_second'] / single
        row['linear_scaling_efficiency'] = row['decode_speedup_vs_single'] / row['batch']
    return result
