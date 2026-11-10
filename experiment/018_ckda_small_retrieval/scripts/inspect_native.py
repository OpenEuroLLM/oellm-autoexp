#!/usr/bin/env python3
"""Inspect native torch-save metadata without importing torch or loading tensors."""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import pickle
import zipfile


class Opaque:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class MetadataReader(pickle.Unpickler):
    def persistent_load(self, pid):
        return Opaque()

    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return OrderedDict
        if (module, name) == ('argparse', 'Namespace'):
            return argparse.Namespace
        # No imported foreign code is executed, including tensor rebuilds.
        return Opaque


def inspect(path):
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith('/data.pkl'))
        with archive.open(name) as f:
            data = MetadataReader(f).load()
    args = vars(data['args'])
    keep = ['num_layers', 'hidden_size', 'ffn_hidden_size', 'num_attention_heads',
            'kv_channels', 'seq_length', 'max_position_embeddings', 'padded_vocab_size',
            'normalization', 'layernorm_epsilon', 'swiglu', 'add_bias_linear',
            'add_qkv_bias', 'qk_layernorm', 'position_embedding_type',
            'untie_embeddings_and_output_weights', 'tensor_model_parallel_size',
            'pipeline_model_parallel_size', 'spec', 'tokenizer_model', 'tokenizer_type',
            'lr', 'min_lr', 'adam_beta1', 'adam_beta2', 'adam_eps', 'weight_decay',
            'global_batch_size', 'micro_batch_size', 'bf16', 'fp16',
            'use_distributed_optimizer', 'train_data_path', 'valid_data_path',
            'consumed_train_samples', 'seed']
    return {'checkpoint_version': data.get('checkpoint_version'),
            'iteration': data.get('iteration'), 'model_keys': len(data['model']),
            'args': {k: args[k] for k in keep if k in args},
            'key_examples': list(data['model'])[:12]}


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    result = inspect(a.checkpoint)
    text = json.dumps(result, indent=2, default=lambda _: '<opaque>') + '\n'
    if a.out:
        a.out.write_text(text)
    print(text)
