"""Verify one child-process CUDA capture records all four GPUs."""
import multiprocessing as mp


def worker(rank, barrier):
    import torch
    torch.cuda.set_device(rank)
    x = torch.randn(256, 256, device='cuda')
    for _ in range(5):
        x = x @ x
    torch.cuda.synchronize()
    barrier.wait()
    if rank == 0:
        torch.cuda.profiler.start()
    barrier.wait()
    for _ in range(16):
        torch.cuda.nvtx.range_push(f'SMOKE/gpu{rank}')
        x = x @ x
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    barrier.wait()
    if rank == 0:
        torch.cuda.profiler.stop()
    barrier.wait()


if __name__ == '__main__':
    ctx = mp.get_context('spawn')
    barrier = ctx.Barrier(4)
    procs = [ctx.Process(target=worker, args=(r, barrier)) for r in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert all(p.exitcode == 0 for p in procs)
