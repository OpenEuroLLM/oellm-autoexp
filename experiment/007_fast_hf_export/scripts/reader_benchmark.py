"""Controlled AB/BA reads on the same node; conversion math is not involved."""
import gc
import importlib.util
import json
import os
from pathlib import Path
import time

import torch
from oellm_autoexp.hf_export import fsdp_dtensor as fast
from oellm_autoexp.hf_export.export_q35 import megatron_args

base = Path('/gpfs/projects/ehpc533/users/korbip/hybrid_v019/autoexp_mhc_eval3700')
spec = importlib.util.spec_from_file_location('legacy_fsdp_reader', base / 'oellm_autoexp/hf_export/fsdp_dtensor.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
ckpt = Path('/gpfs/scratch/ehpc533/hybrid_nucleus/production/main_mhc4/ambitious_e360_eng80_v4_2ph/ckpt_eval/iter_0003700')
config = Path('/gpfs/scratch/ehpc533/hybrid_nucleus/production/main_mhc4/ambitious_e360_eng80_v4_2ph/config-mn5-pool-p129.0338.yaml')
pattern = [c for c in megatron_args(config)['hybrid_layer_pattern'] if c != '|']
selected = {kind: pattern.index(symbol) for kind, symbol in [('attention', '*'), ('ckda', 'L'), ('moe', 'E')]}
readers = {}; startup = {}
for name, module in [('legacy', legacy), ('fast', fast)]:
    os.environ['HF_EXPORT_READER'] = name
    started = time.perf_counter(); readers[name] = module.model_keys(ckpt)
    startup[name] = time.perf_counter() - started
rows = []
for kind, layer in selected.items():
    entries = {k: v for k, v in readers['legacy'][1].items() if k.startswith(f'decoder.layers.{layer}.')}
    for order in [('legacy', 'fast'), ('fast', 'legacy')]:
        loaded = {}; timing = {}
        for name in order:
            os.environ['HF_EXPORT_READER'] = name
            started = time.perf_counter()
            loaded[name] = (legacy if name == 'legacy' else fast).load(readers[name][0], entries)
            timing[name] = time.perf_counter() - started
        assert loaded['legacy'].keys() == loaded['fast'].keys()
        for k, a in loaded['legacy'].items():
            b = loaded['fast'][k]
            assert a.dtype == b.dtype and a.shape == b.shape
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), k
        row = {'kind': kind, 'layer': layer, 'order': order, 'seconds': timing, 'tensors': len(entries), 'bitwise_equal': True}
        rows.append(row);print(json.dumps(row), flush=True)
        del loaded;gc.collect()
result = {'startup_seconds': startup, 'trials': rows, 'note': 'Same node, same input, AB/BA order; no global cache flush. Engram tables covered by full-export comparison.'}
Path('experiment/007_fast_hf_export/results/reader_benchmark.json').write_text(json.dumps(result, indent=2) + '\n')
os.environ['HF_EXPORT_READER'] = 'fast'
