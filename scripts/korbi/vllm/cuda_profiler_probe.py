"""Named worker methods: bracket a region with cudaProfilerStart / Stop (nsys --capture-range=cudaProfilerApi)."""


class CudaProfilerProbe:
    def prof_start(self):
        import torch
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        return True

    def prof_stop(self):
        import torch
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        return True
