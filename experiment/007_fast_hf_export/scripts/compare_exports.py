"""Stream every safetensors tensor and compare raw bytes, independent of shard layout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time

from safetensors import safe_open
import torch


def compare(a, b, workers=4, chunk_bytes=32 * 2**20):
    a, b = Path(a), Path(b)
    ia = json.loads((a/'model.safetensors.index.json').read_text())['weight_map']
    ib = json.loads((b/'model.safetensors.index.json').read_text())['weight_map']
    assert ia.keys() == ib.keys(), 'Tensor names differ'
    groups = {}
    for k in sorted(ia):groups.setdefault((ia[k], ib[k]), []).append(k)
    def group(item):
        (fa, fb), keys = item;total = 0
        with safe_open(a/fa, framework='pt', device='cpu') as x, safe_open(b/fb, framework='pt', device='cpu') as y:
            for key in keys:
                sx, sy = x.get_slice(key), y.get_slice(key)
                shape = sx.get_shape()
                assert shape == sy.get_shape() and sx.get_dtype() == sy.get_dtype(), key
                if not shape:
                    tx, ty = x.get_tensor(key), y.get_tensor(key)
                    assert torch.equal(tx.reshape(-1).view(torch.uint8), ty.reshape(-1).view(torch.uint8)), key
                    total += tx.element_size();continue
                if shape[0] == 0:continue
                element_size = sx[:1].element_size()
                rows = max(1, chunk_bytes // (element_size * math.prod(shape[1:])))
                for start in range(0, shape[0], rows):
                    tx, ty = sx[start:start+rows], sy[start:start+rows]
                    assert torch.equal(tx.contiguous().view(torch.uint8), ty.contiguous().view(torch.uint8)), f'{key} rows {start}'
                total += math.prod(shape) * element_size
        return len(keys), total
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:counts=list(pool.map(group, groups.items()))
    for name in ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json', 'tokenizer.model', 'modeling_oellm_ambitious.py', 'configuration_oellm_ambitious.py', 'modeling_oellm_hybrid.py', 'configuration_oellm_hybrid.py']:
        if (a/name).exists() or (b/name).exists():
            assert (a/name).read_bytes() == (b/name).read_bytes(), name
    return {'tensors':sum(n for n,_ in counts),'tensor_bytes':sum(n for _,n in counts),'bitwise_equal':True,'comparison_seconds':time.perf_counter()-started}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('baseline');p.add_argument('candidate');p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    result=compare(args.baseline,args.candidate,args.workers)
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
