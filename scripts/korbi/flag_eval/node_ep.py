"""Node-local EP: vLLM's DP frontends plus lm-eval's distributed request splitting."""
import multiprocessing as mp
import os
import socket
import time


def _worker(rank, size, evaluator_port, engine_port, argv):
    os.environ.update(FLAG_DP_RANK=str(rank), VLLM_DP_RANK=str(rank),
                      VLLM_DP_RANK_LOCAL=str(rank), VLLM_DP_SIZE=str(size),
                      VLLM_DP_MASTER_IP='127.0.0.1', VLLM_DP_MASTER_PORT=str(engine_port))
    import torch.distributed as dist
    from datetime import timedelta
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{evaluator_port}',
                            rank=rank,world_size=size,timeout=timedelta(minutes=30))
    try:
        from scripts.korbi.flag_eval.flag_lmeval import main
        main(argv)
    finally:
        dist.destroy_process_group()


def launch(size, argv):
    # Reserve both ports together so they cannot accidentally coincide.
    with socket.socket() as one, socket.socket() as two:
        one.bind(('127.0.0.1',0));two.bind(('127.0.0.1',0))
        evaluator_port=one.getsockname()[1];engine_port=two.getsockname()[1]
    ctx=mp.get_context('spawn')
    processes=[ctx.Process(target=_worker,args=(r,size,evaluator_port,engine_port,argv)) for r in range(size)]
    for process in processes:process.start()
    while any(p.is_alive() for p in processes):
        failed=[p for p in processes if p.exitcode not in (None,0)]
        if failed:
            for process in processes:
                if process.is_alive():process.terminate()
            for process in processes:process.join(timeout=5)
            raise RuntimeError(f'EP frontend failure: {[(p.pid,p.exitcode) for p in failed]}')
        time.sleep(0.2)
    for process in processes:process.join()
    if any(p.exitcode for p in processes):raise RuntimeError('EP frontend failed')


def distributed_adapter(base):
    class NodeEPVLLM(base):
        def all_gather(self, tensor):
            import torch
            import torch.distributed as dist
            output=[torch.empty_like(tensor) for _ in range(self.world_size)]
            dist.all_gather(output,tensor)
            return torch.cat([x.reshape(1) if x.ndim==0 else x for x in output])

        def gather_object(self, obj, dst=0):
            import torch.distributed as dist
            output=[None]*self.world_size if self.rank==dst else None
            dist.gather_object(obj,output,dst=dst)
            return output

        def barrier(self):
            import torch.distributed as dist
            dist.barrier()

    return NodeEPVLLM
