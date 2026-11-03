"""Same fixed prompt workload: TP4+EP or TP1+DP4+EP4 on a single node."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import time


def worker(rank, dp, tp, port, args, prompts, barrier):
    if dp > 1:
        os.environ.update(VLLM_DP_RANK=str(rank), VLLM_DP_RANK_LOCAL=str(rank),
                          VLLM_DP_SIZE=str(dp), VLLM_DP_MASTER_IP='127.0.0.1',
                          VLLM_DP_MASTER_PORT=str(port))
    import torch
    from vllm import LLM, SamplingParams
    selected = list(enumerate(prompts))[rank::dp]
    records = [{'prompt_token_ids': p} for _, p in selected]
    started = time.perf_counter()
    # Same explicit multiprocess DP pattern as vLLM's data_parallel_offline.py:
    # DP size/rank are supplied via env, not unsupported single-process LLM(dp=N).
    llm = LLM(model=args.model, trust_remote_code=True, dtype='bfloat16',
              tensor_parallel_size=tp, enable_expert_parallel=True,
              all2all_backend='allgather_reducescatter',
              gpu_memory_utilization=0.85, max_num_seqs=max(1, 32//dp),
              max_model_len=4096, max_num_batched_tokens=512,
              enforce_eager=True, seed=42)
    startup = time.perf_counter()-started
    barrier.wait(timeout=1800)
    sp = SamplingParams(temperature=0, max_tokens=args.decode, prompt_logprobs=0,
                        ignore_eos=True)
    # Separate first-use compilation/page-fault latency from warmed execution.
    trials = []
    for trial in range(2):
        barrier.wait(timeout=1800)
        started = time.perf_counter()
        outputs = llm.generate(records, sp, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        rows = []
        for (idx, ids), output in zip(selected, outputs, strict=True):
            lp = [None if item is None else item[ids[i]].logprob
                  for i, item in enumerate(output.prompt_logprobs)]
            rows.append({'index': idx, 'prompt_token_ids': ids,
                         'prompt_logprobs': lp, 'generated': output.outputs[0].token_ids})
        trials.append({'trial':trial, 'seconds':elapsed, 'rows':rows})
        print(json.dumps({'layout':args.layout,'rank':rank,'trial':trial,
                          'seconds':elapsed,'prompts':len(selected)}),flush=True)
    out = Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/f'rank{rank}.json').write_text(json.dumps({'startup_seconds':startup,
        'tp':tp,'dp':dp,'rank':rank,'trials':trials},indent=2)+'\n')
    barrier.wait(timeout=1800)
    llm.llm_engine.engine_core.shutdown()


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--out',required=True);p.add_argument('--layout',choices=['tp4','ep4'],required=True)
    p.add_argument('--decode',type=int,default=8);p.add_argument('--prompts',type=int,default=32)
    args=p.parse_args()
    from oellm_autoexp.hf_export.score_on_flag import sample_docs
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    mix=out/'datamix.txt'
    mix.write_text('1.0 /gpfs/projects/ehpc533/users/korbip/data/v8ml-valid/all_text_document\n')
    docs=sample_docs(mix,args.prompts,512,0)
    prompts=[[1]+doc[:255].tolist() for doc in docs]
    (out/'prompts.json').write_text(json.dumps(prompts)+'\n')
    dp,tp=(1,4) if args.layout=='tp4' else (4,1)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    ctx=mp.get_context('spawn');barrier=ctx.Barrier(dp)
    processes=[ctx.Process(target=worker,args=(r,dp,tp,port,args,prompts,barrier)) for r in range(dp)]
    for process in processes:process.start()
    for process in processes:process.join(timeout=3600)
    bad=[(x.pid,x.exitcode) for x in processes if x.exitcode!=0]
    for process in processes:
        if process.is_alive():process.kill()
    if bad:raise RuntimeError(f'worker failures: {bad}')
    results=[json.loads((out/f'rank{r}.json').read_text()) for r in range(dp)]
    summary={'layout':args.layout,'tp':tp,'dp':dp,'expert_parallel':4,
             'startup_seconds':max(r['startup_seconds'] for r in results),'trials':[]}
    for trial in range(2):
        elapsed=max(r['trials'][trial]['seconds'] for r in results)
        summary['trials'].append({'seconds':elapsed,'input_tokens':sum(map(len,prompts)),
            'output_tokens':len(prompts)*args.decode,
            'input_tokens_per_second':sum(map(len,prompts))/elapsed})
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
