"""Paired eager/fused/graph controls on the same loaded distributed model."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import time
import traceback


def save(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def worker(rank,args,port,barrier):
    dp=4
    os.environ.update(VLLM_DP_RANK=str(rank),VLLM_DP_RANK_LOCAL=str(rank),VLLM_DP_SIZE=str(dp),
        VLLM_DP_MASTER_IP='127.0.0.1',VLLM_DP_MASTER_PORT=str(port))
    from vllm import LLM,SamplingParams
    all_prompts=json.loads(Path(args.prompts).read_text())
    result={'rank':rank,'trials':[]}
    path=Path(args.out)/f'rank{rank}.json'
    start=time.perf_counter()
    llm=LLM(model=args.model,trust_remote_code=True,dtype='bfloat16',tensor_parallel_size=1,
        enable_expert_parallel=True,all2all_backend='allgather_reducescatter',gpu_memory_utilization=.85,
        max_num_seqs=64,max_num_batched_tokens=8192,max_model_len=4096,enforce_eager=True,
        enable_prefix_caching=False,disable_log_stats=False,seed=42,
        worker_extension_cls='scripts.korbi.vllm.fusion_probe.FusionProbe')
    result['startup_seconds']=time.perf_counter()-start
    result['ckda_checks']=llm.collective_rpc('check_ckda_graphs',timeout=600)
    save(path,result)
    for variant,fused,graphs in [('reference0',False,False),('reference1',False,False),
                                 ('fused',True,False),('graphs',True,True),('graphs_repeat',True,True)]:
        llm.collective_rpc('configure_fusion',timeout=60,kwargs={'fused':fused,'graphs':graphs})
        # Scoring parity plus prefill performance on identical request batches.
        barrier.wait(timeout=1800)
        scoring=[{'prompt_token_ids':p} for p in all_prompts[:128][rank::dp]]
        params=SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=0,ignore_eos=True)
        llm.generate(scoring[:4],params,use_tqdm=False)
        barrier.wait(timeout=1800)
        start=time.perf_counter();outputs=llm.generate(scoring,params,use_tqdm=False)
        scores=[]
        for output in outputs:
            values=[]
            for token,record in zip(output.prompt_token_ids[1:],output.prompt_logprobs[1:]):
                values.append(record[token].logprob)
            scores.append(values)
        result['trials'].append({'variant':variant,'mode':'scoring','seconds':time.perf_counter()-start,'scores':scores})
        save(path,result)
        for batch in [1,32,256]:
            chosen=all_prompts[:batch][rank::dp]
            dummy=not chosen
            work=[{'prompt_token_ids':p} for p in (chosen or all_prompts[:1])]
            barrier.wait(timeout=1800)
            llm.generate(work,SamplingParams(temperature=0,max_tokens=1 if dummy else 32,ignore_eos=True),use_tqdm=False)
            barrier.wait(timeout=1800)
            start=time.perf_counter()
            outputs=llm.generate(work,SamplingParams(temperature=0,max_tokens=1 if dummy else 128,ignore_eos=True),use_tqdm=False)
            entry={'variant':variant,'mode':'decode','batch':batch,'wall_seconds':time.perf_counter()-start,'requests':[]}
            if not dummy:
                for output in outputs:
                    assert len(output.outputs[0].token_ids)==128
                    m=output.metrics
                    entry['requests'].append({'tokens':output.outputs[0].token_ids,
                        'first_token_ts':m.first_token_ts,'last_token_ts':m.last_token_ts})
            entry['graphs']=llm.collective_rpc('graph_stats',timeout=60)
            entry['memory']=llm.collective_rpc('batching_memory',timeout=60)
            result['trials'].append(entry);save(path,result)
            print(json.dumps({'variant':variant,'rank':rank,'batch':batch,'wall_seconds':entry['wall_seconds'],'graphs':entry['graphs']}),flush=True)
    barrier.wait(timeout=1800)
    llm.llm_engine.engine_core.shutdown()


def entry(*args):
    try: worker(*args)
    except BaseException:
        print(traceback.format_exc(),flush=True);os._exit(1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--prompts',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();Path(args.out).mkdir(parents=True,exist_ok=True)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    ctx=mp.get_context('spawn');barrier=ctx.Barrier(4)
    procs=[ctx.Process(target=entry,args=(rank,args,port,barrier)) for rank in range(4)]
    for proc in procs:proc.start()
    try:
        while any(p.is_alive() for p in procs):
            if any(p.exitcode not in (None,0) for p in procs):raise RuntimeError('Distributed benchmark worker failed')
            time.sleep(.5)
        assert all(p.exitcode==0 for p in procs)
    finally:
        for proc in procs:
            if proc.is_alive():proc.terminate()
        for proc in procs:proc.join(timeout=5)

if __name__=='__main__':main()
