"""Scoped inference diagnostics over the original signed-KDA implementation."""


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
