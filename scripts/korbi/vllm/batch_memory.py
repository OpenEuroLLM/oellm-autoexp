"""Named worker RPC for benchmark memory measurements, without function serialization."""


class BatchMemoryProbe:
    def batching_memory(self, reset=False):
        import os
        import torch

        if reset:
            torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        return {'pid': os.getpid(), 'parent_pid': os.getppid(), 'device': torch.cuda.current_device(), 'free_bytes': free,
                'total_bytes': total, 'allocated_bytes': torch.cuda.memory_allocated(),
                'reserved_bytes': torch.cuda.memory_reserved(),
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved()}

    def check_batched_engram(self):
        import importlib

        check = importlib.import_module(
            'experiment.008_ambitious_inference_speed.scripts.check_batched_engram')
        return check.check_modules(self.model_runner.model)

    def configure_engram_decode(self, batched=False):
        from vllm.model_executor.models.oellm_ambitious import OellmEngram

        count = 0
        for module in self.model_runner.model.modules():
            if isinstance(module, OellmEngram):
                module.batched_decode = batched
                count += 1
        return count
