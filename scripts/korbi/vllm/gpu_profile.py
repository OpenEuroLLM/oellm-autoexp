"""Named worker methods for bounded, annotated Nsight decode captures."""
from scripts.korbi.vllm.batch_memory import BatchMemoryProbe


class DecodeTimelineProbe(BatchMemoryProbe):
    def arm_gpu_trace(self, label, skip=8, steps=12):
        import functools
        import os
        import torch
        from vllm.distributed import get_ep_group
        from vllm.forward_context import get_forward_context

        assert not getattr(self, '_timeline_handles', []), 'Disarm previous capture first'
        model = self.model_runner.model
        self._timeline_handles = []
        self._timeline_restore = []
        self._timeline_label = label
        self._timeline_seen = 0
        self._timeline_active = False
        self._timeline_completed = False
        self._timeline_control = get_ep_group().rank_in_group == 0
        rank = get_ep_group().rank_in_group
        prefixes = [m.prefix for m in model.modules() if type(m).__name__ == 'OellmEngram']
        assert prefixes
        prefix = prefixes[0]

        def root_pre(module, args):
            raw = get_forward_context().attn_metadata
            meta = raw.get(prefix) if isinstance(raw, dict) else None
            decode = meta is not None and meta.num_prefills == 0 and meta.num_decode_tokens > 0
            if decode:
                self._timeline_seen += 1
            count = self._timeline_seen
            if self._timeline_control and decode and count == skip + 1:
                torch.cuda.synchronize()
                torch.cuda.profiler.start()
                self._timeline_active = True
            if self._timeline_control and decode and count == skip + steps + 1:
                torch.cuda.synchronize()
                torch.cuda.nvtx.mark('CAPTURE_END/' + label)
                torch.cuda.profiler.stop()
                self._timeline_active = False
                self._timeline_completed = True
            phase = f'decode{count}/rows{meta.num_decode_tokens}' if decode else 'prefill_or_dummy'
            torch.cuda.nvtx.range_push(f'STEP/{label}/r{rank}/{phase}')

        def root_post(module, args, output):
            torch.cuda.nvtx.range_pop()

        self._timeline_handles += [model.register_forward_pre_hook(root_pre),
                                   model.register_forward_hook(root_post, always_call=True)]

        def make_pre(text):
            def pre(module, args):
                torch.cuda.nvtx.range_push(text)
            return pre

        def post(module, args, output):
            torch.cuda.nvtx.range_pop()

        def wrap_method(original, text):
            @functools.wraps(original)
            def call(*args, **kwargs):
                torch.cuda.nvtx.range_push(text)
                try:
                    return original(*args, **kwargs)
                finally:
                    torch.cuda.nvtx.range_pop()
            return call

        classes = {'OellmRMSNorm', 'OellmCKDA', 'OellmMoE', 'OellmEngram',
                   'OellmXSAAttention', 'OellmLDMRead', 'OellmLDMWrite'}
        for name, module in model.named_modules():
            cls = type(module).__name__
            if cls in classes:
                text = f'MODULE/{cls}/{name}'
                self._timeline_handles += [module.register_forward_pre_hook(make_pre(text)),
                                           module.register_forward_hook(post, always_call=True)]
            if cls == 'OellmMHC':
                for method in ['read', 'write']:
                    original = getattr(module, method)
                    self._timeline_restore.append((module, method, original))
                    setattr(module, method, wrap_method(original, f'METHOD/MHC.{method}/{name}'))
        return {'pid': os.getpid(), 'rank': rank, 'controller': self._timeline_control,
                'label': label, 'skip_decode_steps': skip, 'capture_decode_steps': steps}

    def disarm_gpu_trace(self):
        import os
        import torch

        for handle in self._timeline_handles:
            handle.remove()
        for module, name, original in self._timeline_restore:
            setattr(module, name, original)
        if self._timeline_active:
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
        self._timeline_handles = []
        self._timeline_restore = []
        return {'pid': os.getpid(), 'label': self._timeline_label,
                'decode_steps_seen': self._timeline_seen,
                'controller': self._timeline_control,
                'capture_completed': self._timeline_completed}
