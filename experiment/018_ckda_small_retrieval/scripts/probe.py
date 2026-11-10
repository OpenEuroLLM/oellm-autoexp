#!/usr/bin/env python3
"""Check original model kernels and offline task availability on one GPU."""
import json
import os
from pathlib import Path
import sys

root = Path(os.environ['CKDA_WORK'])
sys.path[:0] = [str(root / 'source'), str(root / 'source/lm_scaling')]
from eval_downstream import ensure_lm_eval
ensure_lm_eval()
from metric_compat import install
install()
import torch
import transformers
import lm_eval
from fla.models.signed_kda import SignedKDAForCausalLM
from fla.models.signed_kda.configuration_signed_kda import SignedKDAConfig

cfg = SignedKDAConfig(hidden_size=256, num_hidden_layers=4, num_heads=4,
                     head_dim=64, intermediate_size=512, hidden_ratio=None,
                     vocab_size=256, max_position_embeddings=4096,
                     tie_word_embeddings=False, gate='sigmoid',
                     allow_neg_eigval=True, output_gate='lowrank', drop_silu=True,
                     attn={'layers': [3], 'num_heads': 4, 'num_kv_heads': 4,
                           'qkv_bias': False, 'qk_norm': False,
                           'output_gate': True, 'use_rope': False,
                           'window_size': None, 'rope_theta': 10000.0})
model = SignedKDAForCausalLM(cfg).to('cuda', torch.bfloat16).eval()
ids = torch.arange(128, device='cuda').unsqueeze(0)
with torch.no_grad():
    out = model(ids, use_cache=False).logits
assert out.shape == (1, 128, 256) and torch.isfinite(out).all()
with torch.no_grad():
    full = model(ids[:, :65], use_cache=False).logits[:, -1]
    prefix = model(ids[:, :64], use_cache=True)
    cached = model(ids[:, 64:65], past_key_values=prefix.past_key_values,
                   use_cache=True).logits[:, -1]
torch.testing.assert_close(cached, full, rtol=0.02, atol=0.03)
print('CHUNK_CACHE_PARITY_PASS', flush=True)
# A bounded KDA control must remain exactly unchanged by removing signed behavior.
cfg.allow_neg_eigval = False
control = SignedKDAForCausalLM(cfg).to('cuda', torch.bfloat16).eval()
from interventions import Intervention
with torch.no_grad():
    expected = control(ids, use_cache=False).logits
    for condition in ['no_reflection', 'no_neg_alpha', 'both',
                      'erase_cap_preserve_write', 'write_cap_preserve_erase']:
        with Intervention(control, condition) as patch:
            actual = control(ids, use_cache=False).logits
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert patch.stats['calls'] > 0
print('KDA_SIGN_CONTROLS_PASS', flush=True)
print(json.dumps({'torch': torch.__version__, 'transformers': transformers.__version__,
                  'lm_eval': getattr(lm_eval, '__version__', 'unknown'),
                  'device': torch.cuda.get_device_name(), 'logits': list(out.shape)}), flush=True)
from lm_eval.tasks import TaskManager, get_task_dict
tm = TaskManager(include_path=str(root / 'source/lm_scaling/eval_tasks'))
errors = []
for task in ['winogrande', 'squadv2', 'coqa', 'squad_completion', 'swde', 'fda']:
    try:
        obj = get_task_dict([task], tm)[task]
        docs = obj.validation_docs() if obj.has_validation_docs() else obj.test_docs()
        print('TASK_READY', task, len(docs), flush=True)
    except Exception as exc:
        print('TASK_ERROR', task, repr(exc), flush=True)
        errors.append((task, repr(exc)))
if errors:
    raise RuntimeError(errors)
(root / 'results/probe_pass.json').write_text(json.dumps({
    'torch': torch.__version__, 'transformers': transformers.__version__,
    'device': torch.cuda.get_device_name(), 'tasks': 'all six ready'}) + '\n')
print('PROBE_PASS', flush=True)
