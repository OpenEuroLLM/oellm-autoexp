"""Named worker RPC for benchmark memory measurements, without function serialization."""


class BatchMemoryProbe:
    def batching_memory(self, reset=False):
        import torch

        if reset:
            torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        return {'device': torch.cuda.current_device(), 'free_bytes': free,
                'total_bytes': total, 'allocated_bytes': torch.cuda.memory_allocated(),
                'reserved_bytes': torch.cuda.memory_reserved(),
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
