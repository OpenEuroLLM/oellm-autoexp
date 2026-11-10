"""Scoped inference diagnostics over the original signed-KDA implementation."""


def split_beta_patch(module, condition, stats):
    """Separate beta's erase/write effects at the kernel entry point.

    The kernel uses beta * k * (v - S.T @ k). Rescaling v lets us change
    writing independently of the beta multiplying the erased state. This
    preserves the intended term algebraically, up to v's bf16 rounding.
    """
    import torch
    originals = {}
    for name in ['chunk_kda', 'fused_recurrent_kda']:
        original = getattr(module, name, None)
        if original is None:
            continue
        originals[name] = original

        def wrap(original):
            def call(*args, **kwargs):
                if kwargs.get('beta') is not None and kwargs.get('k') is not None:
                    stats['calls'] += 1
                    if kwargs.get('allow_neg_eigval', False):
                        if not kwargs.get('use_beta_sigmoid_in_kernel'):
                            raise ValueError('Expected beta logits with the kernel sigmoid')
                        logits = kwargs['beta']
                        beta = 2 * torch.sigmoid(logits.float())
                        capped_logits = logits.clamp(max=0)
                        capped = 2 * torch.sigmoid(capped_logits.float())
                        if condition == 'erase_cap_preserve_write':
                            scale = torch.where(beta > 1, beta / capped, 1.0)
                            kwargs['beta'] = capped_logits
                        else:
                            scale = torch.where(beta > 1, capped / beta, 1.0)
                        value = kwargs['v']
                        kwargs['v'] = (value.float() * scale.unsqueeze(-1)).to(value.dtype)
                        stats['beta_gt1'] += float((beta > 1).float().mean())
                return original(*args, **kwargs)
            return call
        setattr(module, name, wrap(original))
    return originals


class Intervention:
    def __init__(self, model, condition):
        self.model, self.condition = model, condition
        self.stats = {'condition': condition, 'calls': 0, 'beta_gt1': 0.0,
                      'alpha_neg': 0.0, 'g_mean': 0.0}
        self.originals, self.windows = {}, []

    def __enter__(self):
        if self.condition == 'baseline':
            return self
        if self.condition.startswith('window_'):
            from fla.layers.attn import Attention
            for layer in self.model.model.layers:
                if isinstance(layer.attn, Attention):
                    self.windows.append((layer.attn, layer.attn.window_size))
                    layer.attn.window_size = int(self.condition.split('_')[1])
            if not self.windows:
                raise ValueError('No attention layers patched')
        else:
            import fla.layers.signed_kda_layer as skda
            from recall_intervene import _patch_gate
            if self.condition in ['erase_cap_preserve_write', 'write_cap_preserve_erase']:
                self.originals = split_beta_patch(skda, self.condition, self.stats)
            else:
                self.originals = _patch_gate(skda, self.condition, self.stats)
            if not self.originals:
                raise ValueError('No KDA kernel entry points patched')
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.originals:
            import fla.layers.signed_kda_layer as skda
            from recall_intervene import _restore
            _restore(skda, self.originals)
        for layer, window in self.windows:
            layer.window_size = window
        if exc_type is None and self.originals and not self.stats['calls']:
            raise RuntimeError('Intervention was installed but never reached a kernel')
